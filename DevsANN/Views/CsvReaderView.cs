using DevsANN.Data;
using DevsANN.Models;
using DEVSsharp;
using System.Collections.Generic;
using System.IO;

namespace DevsANN.Views
{
    class CsvReaderView : Atomic
    {
        private StreamReader sr;
        private List<FlowerModel> data;
        private int currentIndex;
        private readonly string trainFilePath;
        private readonly string testFilePath;

        readonly int neuronsToWaitForNextRead;
        int neuronsToWaitCounter;

        int currentEpoch; // the current epoch
        public static int passesThorughEpoch; // epochs / batch_size (+1)
        int currentPassageThroughEpoch;

        public InputPort loadTestDataInputPort; // a port to signal load of training samples
        public InputPort loadTrainingDataInputPort;
        public InputPort fetchNextTestExampleInputPort; // start network training for the given input
        public InputPort fetchNextTrainingExampleInputPort;
        public List<OutputPort> neuronsDataOutputPort; // those ports are used to send data to the neurons in the input layer

        public OutputPort startTestPhaseOutputPort;
        public OutputPort showResultsOutputPort;
        public OutputPort desiredResultOutputPort;
        public OutputPort loadTrainingDataOutputPort;

        private Queue<PortValue> msgBuff;
        public CsvReaderView(string name, TimeUnit tu, string trainFilePath, string testFilePath, int neuronsToWaitForNextRead) : base(name, tu)
        {
            this.trainFilePath = trainFilePath;
            this.testFilePath = testFilePath;
            this.neuronsToWaitForNextRead = neuronsToWaitForNextRead;

            loadTestDataInputPort = AddIP(name + "_loadTestDataInputPort");
            loadTrainingDataInputPort = AddIP(name + "_loadTrainingDataInputPort");
            loadTrainingDataOutputPort = AddOP(name + "_loadTrainingDataOutputPort");
            fetchNextTrainingExampleInputPort = AddIP(name + "_fetchNextTrainingExampleInputPort");
            fetchNextTestExampleInputPort = AddIP(name + "_fetchNextTestExampleInputPort");
            startTestPhaseOutputPort = AddOP("startTestPhaseOutputPort");
            showResultsOutputPort = AddOP("showResultsOutputPort");
            desiredResultOutputPort = AddOP(name + "_desiredResultOutputPort");

            neuronsDataOutputPort = new List<OutputPort>();
            for (int i = 0; i < Config.INPUT_LAYER_NEURONS_COUNT; i++)
                neuronsDataOutputPort.Add(AddOP("neuronsDataOutputPort" + i + ""));

            init();
        }
        public override bool delta_x(PortValue x)
        {
            if (x.port == loadTrainingDataInputPort)
            {
                LoadData(trainFilePath, false);
                currentIndex = 0;

                msgBuff.Enqueue(new PortValue(desiredResultOutputPort, data[currentIndex].FlowerType));
                for (int i = 0; i < neuronsDataOutputPort.Count; i++)
                    msgBuff.Enqueue(new PortValue(neuronsDataOutputPort[i], data[currentIndex].GetDataByNeuronID(i))); // send data to the neruons in the input layer

                ++currentIndex;
                return true;
            }
            else if (x.port == loadTestDataInputPort)
            {
                LoadData(testFilePath, true);
                currentIndex = 0;
                neuronsToWaitCounter = 0;
                CsvWriterView.testPhase = true;

                msgBuff.Enqueue(new PortValue(desiredResultOutputPort, data[currentIndex].FlowerType));
                for (int i = 0; i < neuronsDataOutputPort.Count; i++)
                    msgBuff.Enqueue(new PortValue(neuronsDataOutputPort[i], data[currentIndex].GetDataByNeuronID(i)));

                ++currentIndex;
                return true;
            }
            else if (x.port == fetchNextTrainingExampleInputPort)
            {
                // ----------- wait for all neurons in the previous layer to generate their outputs
                ++neuronsToWaitCounter;
                if (neuronsToWaitCounter < neuronsToWaitForNextRead)
                    return false;
                else
                    neuronsToWaitCounter = 0;

                if (currentIndex < data.Count)
                {
                    msgBuff.Enqueue(new PortValue(desiredResultOutputPort, data[currentIndex].FlowerType));
                    for (int i = 0; i < neuronsDataOutputPort.Count; i++)
                        msgBuff.Enqueue(new PortValue(neuronsDataOutputPort[i], data[currentIndex].GetDataByNeuronID(i)));

                    currentIndex++;
                }
                else // all data from the batch are read
                {
                    ++currentPassageThroughEpoch; // increase the number of batches done so far per epoch

                    if (currentPassageThroughEpoch < passesThorughEpoch) // load the next batch_size
                        msgBuff.Enqueue(new PortValue(loadTrainingDataOutputPort, null));
                }

                if (currentPassageThroughEpoch == passesThorughEpoch) // one epoch is completed
                {
                    ++currentEpoch;
                    sr.Close();

                    if (currentEpoch == Config.EPOCHS) // all epoches are done
                        msgBuff.Enqueue(new PortValue(startTestPhaseOutputPort, null));
                    else
                    {
                        sr = new StreamReader(Config.TRAIN_FILE_PATH);
                        sr.ReadLine();
                        currentPassageThroughEpoch = 0; // reset the number of finished batches because we are moving to the next epoch
                        msgBuff.Enqueue(new PortValue(loadTrainingDataOutputPort, null));
                    }
                }

                return true;
            }
            else if (x.port == fetchNextTestExampleInputPort)
            {
                if (currentIndex < data.Count)
                {
                    msgBuff.Enqueue(new PortValue(desiredResultOutputPort, data[currentIndex].FlowerType));
                    for (int i = 0; i < neuronsDataOutputPort.Count; i++)
                        msgBuff.Enqueue(new PortValue(neuronsDataOutputPort[i], data[currentIndex].GetDataByNeuronID(i)));

                    ++currentIndex;
                    return true;
                }
                else
                {
                    msgBuff.Enqueue(new PortValue(showResultsOutputPort, null)); // konacan rezultat
                    return true;
                }
            }

            return false;
        }

        public override void delta_y(ref PortValue y)
        {
            if (msgBuff.Count > 0)
                y = msgBuff.Dequeue();
            return;
        }

        public override void init()
        {
            currentEpoch = 0;
            passesThorughEpoch = Config.TRAINING_FILE_SIZE % Config.BATCH_SIZE == 0 ? Config.TRAINING_FILE_SIZE / Config.BATCH_SIZE : Config.TRAINING_FILE_SIZE / Config.BATCH_SIZE + 1;
            currentPassageThroughEpoch = 0;

            neuronsToWaitCounter = 0;

            if (sr != null)
                sr.Close();

            sr = new StreamReader(Config.TRAIN_FILE_PATH);
            sr.ReadLine(); // read header
            data = new List<FlowerModel>();

            currentIndex = 0;
            msgBuff = new Queue<PortValue>();
        }

        public override double tau()
        {
            if (msgBuff.Count > 0)
                return 0.0;

            return double.PositiveInfinity;
        }
        public void LoadData(string filePath, bool testPhase)
        {
            if (testPhase)
            {
                sr.Close();
                sr = new StreamReader(Config.TEST_FILE_PATH);
                sr.ReadLine();
            }

            data.Clear();
            string line;
            string[] parts;
            int readCount = 0; // the count of input training samples

            while ((testPhase || readCount < Config.BATCH_SIZE) && (line = sr.ReadLine()) != null) // if it is a testing phase, then batch end is not important
            {
                ++readCount;
                parts = line.Split(',');
                data.Add(new FlowerModel
                {
                    SepalLength = double.Parse(parts[0]),
                    SepalWidth = double.Parse(parts[1]),
                    PetalLength = double.Parse(parts[2]),
                    PetalWidth = double.Parse(parts[3]),
                    FlowerType = int.Parse(parts[4])
                });
            }
        }

        public override string Get_s()
        {
            return "EPOCH: " + currentEpoch + " / " + Config.EPOCHS + " *** BATCH: " + currentPassageThroughEpoch + " / " + passesThorughEpoch + " *** DATA: " + currentIndex + " / " + (data != null ? data.Count : 0) + "";
        }

    }
}
