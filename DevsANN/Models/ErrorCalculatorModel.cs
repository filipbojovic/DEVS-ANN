using DevsANN.Data;
using DevsANN.Views;
using DEVSsharp;
using System.Collections.Generic;
using System.Linq;

namespace DevsANN.Models
{
    class ErrorCalculatorModel : Atomic
    {
        public static int neuronWithCorrectOutput; // a neurown with the correct output
        public int outputValuesCounter; // number of output layer neurons which have sent their outputs so far
        public List<double> outputValues; // generated values by output layer neurons
        int testDataLength;
        int numberOfHits;

        List<List<double>> errorDerivativeValues;
        int dataInsideBatch = 0;
        int readBatches = 0;

        public List<InputPort> neuronOutputInputsPorts; // input ports of neurons in the output layer
        public InputPort desiredOutputInputPort;
        public InputPort calculateAccuracyInputPort;

        public List<OutputPort> errorDerivativeOutputPorts; // derivative dC/dActivation
        public OutputPort accuracyOutputPort; // it is used to tell csvWriter the network accuracy
        public OutputPort fetchNextDataOutputPort; // it is used to tell the csvReader to proceed with reading the next input

        private Queue<PortValue> msgBuff;

        public ErrorCalculatorModel(string name, TimeUnit tu) : base(name, tu)
        {
            desiredOutputInputPort = AddIP(name + "_desiredOutputInputPort");
            calculateAccuracyInputPort = AddIP(name + "_calculateAccuracyInputPort");
            accuracyOutputPort = AddOP(name + "_accuracyOutputPort");
            fetchNextDataOutputPort = AddOP(name + "_fetchNextDataOutputPort");

            neuronOutputInputsPorts = new List<InputPort>();
            for (int i = 0; i < Config.OUTPUT_LAYER_NEURONS_COUNT; i++)
                neuronOutputInputsPorts.Add(AddIP(name + "_outputFromNeuron" + i.ToString()));

            errorDerivativeOutputPorts = new List<OutputPort>();
            for (int i = 0; i < Config.OUTPUT_LAYER_NEURONS_COUNT; i++)
                errorDerivativeOutputPorts.Add(AddOP(name + "_errDer" + i.ToString()));

            init();
        }

        public override bool delta_x(PortValue x)
        {
            if (x.port == desiredOutputInputPort) // the READER sent what is the expected output for the current training exmaple (input)
            {
                neuronWithCorrectOutput = (int)x.value;
                return false;
            }
            else if (x.port == calculateAccuracyInputPort)
            {
                msgBuff.Enqueue(new PortValue(accuracyOutputPort, (double)numberOfHits / testDataLength));
                return true;
            }

            for (int i = 0; i < Config.OUTPUT_LAYER_NEURONS_COUNT; i++)
                if (x.port == neuronOutputInputsPorts[i])
                {
                    if (!CsvWriterView.testPhase)
                    {
                        if (dataInsideBatch == Config.BATCH_SIZE)
                            for (int j = 0; j < Config.OUTPUT_LAYER_NEURONS_COUNT; j++)
                                errorDerivativeValues[j].Clear();

                        double errorDerivative = 2.0 * ((double)x.value - (i == neuronWithCorrectOutput ? 1.0 : 0.0));
                        ++outputValuesCounter;
                        errorDerivativeValues[i].Add(errorDerivative); // add delta error value to the corresponding

                        if (outputValuesCounter == Config.OUTPUT_LAYER_NEURONS_COUNT)
                        {
                            outputValuesCounter = 0;
                            ++dataInsideBatch; // the number of training examples (inputs) already propagated through the network
                            // all answers received (3 of them), check if a "batch end" should occurr now

                            bool temp = (readBatches == CsvReaderView.passesThorughEpoch - 1) && (dataInsideBatch == Config.TRAINING_FILE_SIZE % Config.BATCH_SIZE);

                            if (dataInsideBatch == Config.BATCH_SIZE || temp) // batch end. Start the backpropagation process
                            {
                                ++readBatches;

                                if (readBatches == CsvReaderView.passesThorughEpoch)
                                    readBatches = 0;

                                dataInsideBatch = 0; // number of read data inside a batch
                                for (int j = 0; j < Config.OUTPUT_LAYER_NEURONS_COUNT; j++)
                                    msgBuff.Enqueue(new PortValue(errorDerivativeOutputPorts[j], errorDerivativeValues[j]));
                            }
                            else // if it is not the batch end, then read the following input from the batch
                                msgBuff.Enqueue(new PortValue(fetchNextDataOutputPort, null));

                            return true;
                        }
                        else
                            return false;
                    }
                    else
                    {
                        ++outputValuesCounter;
                        outputValues[i] = (double)x.value;
                        if (outputValuesCounter == Config.OUTPUT_LAYER_NEURONS_COUNT)
                        {
                            outputValuesCounter = 0;

                            double maxValue = outputValues.Max();
                            int index = outputValues.IndexOf(maxValue);

                            ++testDataLength;
                            if (index == neuronWithCorrectOutput)
                                ++numberOfHits;
                            msgBuff.Enqueue(new PortValue(fetchNextDataOutputPort, null));
                            return true;
                        }
                        else
                            return false;
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
            msgBuff = new Queue<PortValue>();
            outputValuesCounter = 0;
            outputValues = new List<double>();
            errorDerivativeValues = new List<List<double>>();
            for (int i = 0; i < Config.OUTPUT_LAYER_NEURONS_COUNT; i++)
            {
                outputValues.Add(0.0);
                errorDerivativeValues.Add(new List<double>());
            }

            testDataLength = 0;
            numberOfHits = 0;
        }

        public override double tau()
        {
            if (msgBuff.Count > 0)
                return 0.0;

            return double.PositiveInfinity;
        }

        public override string Get_s()
        {
            return "Accuracy: " + ((double)numberOfHits / (double)testDataLength * 100.0).ToString();
        }
    }
}
