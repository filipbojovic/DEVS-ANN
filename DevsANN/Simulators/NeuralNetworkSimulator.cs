using DevsANN.Data;
using DevsANN.Models;
using DevsANN.Views;
using DEVSsharp;
using System;
using System.Collections.Generic;

namespace DevsANN.Simulators
{
    class NeuralNetworkSimulator : Coupled
    {
        readonly int total_layers_count;
        public static Random rand = new Random();
        int ind = 0;

        List<ConnectionModel[,]> conns;
        List<LayerModel> layers;
        CsvWriterView csvWriter;
        CsvReaderView csvReader;
        ErrorCalculatorModel errorCalculator;

        public InputPort trainInputPort; // it is used to send a signal to start with loading training data
        public InputPort testInputPort;

        // put the same initial weight values for DevsANN as for a Tensorflow model, to improve a comparasion between them
        List<double> weights = new List<double> { -3.488149669089023, 3.531869688835951, -2.00319871057288, 4.762910457602146, 1.4905973706355464, -2.6829832513703273,
            5.570517789879953, -7.307169745344667, -7.27729926766101, 0.007224552977579561, 3.584015999627222, 5.71913220327202, -1.7093554743996349, -8.217365146262289}; // 93.33%

        public NeuralNetworkSimulator(string name) : base(name)
        {
            trainInputPort = AddIP("trainInputPort");
            testInputPort = AddIP("testInputPort");

            layers = new List<LayerModel>();
            conns = new List<ConnectionModel[,]>();
            total_layers_count = 2 + Config.NUMBER_OF_HIDDEN_LAYERS;

            InstantiateLayers();
            InstantiateConnections();
            csvWriter = new CsvWriterView("csvWriter", TimeUnit.Sec, Config.TRAIN_FILE_PATH, Config.OUTPUT_FILE_PATH);
            csvReader = new CsvReaderView("csvReader", TimeUnit.Sec, Config.TRAIN_FILE_PATH, Config.TEST_FILE_PATH, Config.INPUT_LAYER_NEURONS_COUNT * layers[1].GetNeuronsCount());
            errorCalculator = new ErrorCalculatorModel("errorCalc", TimeUnit.Sec);

            AddModels();
            AddCouplings();
        }
        private void AddCouplings()
        {
            ConnectionModel[,] conn;

            for (int i = 0; i < total_layers_count - 1; i++) // the connection between i-th and (i + 1)th layer corresponds to the i-th connection weight in conns
            {
                conn = conns[i];

                for (int k = 0; k < layers[i].GetNeuronsCount(); k++)
                    for (int l = 0; l < layers[i + 1].GetNeuronsCount(); l++)
                    {
                        AddCP(layers[i].neurons[k].outputPort, conn[k, l].inputNeuronPort);
                        AddCP(conn[k, l].outputNeuronPort, layers[i + 1].neurons[l].inputPorts[k]);
                    }
            }

            // -------------connect csvReader and input layer ---------
            AddCP(trainInputPort, csvReader.loadTrainingDataInputPort); // the user input to start the simulation
            AddCP(csvReader.loadTrainingDataOutputPort, csvReader.loadTrainingDataInputPort); // the automatic load of next BATCH_SIZE training inputs

            for (int i = 0; i < Config.INPUT_LAYER_NEURONS_COUNT; i++)
                AddCP(csvReader.neuronsDataOutputPort[i], layers[0].neurons[i].inputPorts[0]);

            // -------------connect the output layer with the csv
            // ------------------------------------------------------------
            for (int i = 0; i < layers[layers.Count - 1].GetNeuronsCount(); i++)
            {
                AddCP(layers[layers.Count - 1].neurons[i].outputPort, csvWriter.estimatedValuePorts[i]); // tell CSV writer to write values
                AddCP(layers[layers.Count - 1].neurons[i].outputPort, errorCalculator.neuronOutputInputsPorts[i]); // send a signal for error calculation
                AddCP(errorCalculator.errorDerivativeOutputPorts[i], layers[layers.Count - 1].neurons[i].startBackPropInputPort);
            }

            AddCP(csvReader.desiredResultOutputPort, errorCalculator.desiredOutputInputPort); // once a training sample is read, immediatelly tell the output layer what is the expected output

            // -------------- once a value is written into csv, send a signal to neurons in the output layer to start the backpropagation algorithm.
            LayerModel outputLayer = layers[layers.Count - 1];
            for (int i = 0; i < outputLayer.GetNeuronsCount(); i++)
                AddCP(errorCalculator.errorDerivativeOutputPorts[i], outputLayer.neurons[i].startBackPropInputPort);

            // -------------- set up connections for the backpropagation algorithm -------------------
            for (int i = layers.Count - 1; i > 0; i--)
            {
                conn = conns[i - 1];
                for (int j = 0; j < layers[i].GetNeuronsCount(); j++)
                    for (int k = 0; k < layers[i - 1].GetNeuronsCount(); k++)
                    {
                        AddCP(layers[i].neurons[j].deltaErrorOutputPort, conn[k, j].backPropInputPort);
                        AddCP(conn[k, j].backPropOutputPort, (i > 1) ? (layers[i - 1].neurons[k].deltaErrorInputPort) : (csvReader.fetchNextTrainingExampleInputPort)); // ako je poslednji sloj, javi da se ucita sledeci podatak
                    }
            }

            AddCP(testInputPort, csvReader.loadTestDataInputPort); // it is used for a user input to start the testing phase
            AddCP(csvReader.startTestPhaseOutputPort, csvReader.loadTestDataInputPort); // automatic start of the testing phase
            AddCP(errorCalculator.fetchNextDataOutputPort, csvReader.fetchNextTestExampleInputPort); // ERROR MODEL tells the reader to load the next training sample
            AddCP(csvReader.showResultsOutputPort, errorCalculator.calculateAccuracyInputPort); // reader tells ERROR_CALC to calculate the network accuracy
            AddCP(errorCalculator.accuracyOutputPort, csvWriter.showResultsInputPort); // ERROR_CALC outputs the network accuracy
        }

        private void AddModels()
        {
            foreach (var layer in layers)
            {
                AddModel(layer);
                foreach (var item in layer.neurons)
                    AddModel(item);
            }
            foreach (var conn in conns)
                foreach (var item in conn)
                    AddModel(item);
            AddModel(csvWriter);
            AddModel(csvReader);
            AddModel(errorCalculator);
        }

        private void InstantiateLayers()
        {
            int previousLayerNeuronsCount;
            layers.Add(new LayerModel("inputLayer", Config.INPUT_LAYER_NEURONS_COUNT, 1, Config.EPS_Z, Config.TAU_DELAY, Config.HIDDEN_LAYER_NEURONS_COUNT)); // the input layer
            for (int i = 1; i <= Config.NUMBER_OF_HIDDEN_LAYERS; i++) // i = 1 so we can catch the input layer with (i - 1)
            {
                previousLayerNeuronsCount = layers[i - 1].GetNeuronsCount();
                layers.Add(new LayerModel("hiddenLayer" + i + "", Config.HIDDEN_LAYER_NEURONS_COUNT, previousLayerNeuronsCount, Config.EPS_Z, Config.TAU_DELAY, i == Config.NUMBER_OF_HIDDEN_LAYERS ? Config.OUTPUT_LAYER_NEURONS_COUNT : Config.HIDDEN_LAYER_NEURONS_COUNT));
            }
            previousLayerNeuronsCount = layers[Config.NUMBER_OF_HIDDEN_LAYERS].GetNeuronsCount();
            layers.Add(new LayerModel("outputLayer", Config.OUTPUT_LAYER_NEURONS_COUNT, previousLayerNeuronsCount, Config.EPS_Z, Config.TAU_DELAY, 0)); // the output layer; layers[hidden_l_c] gives exactly the output layer.
        }
        private void InstantiateConnections()
        {
            int n, m; // n - the number of neurons in i-th layer; m - the number of neurons in the (i + 1)-th layer.

            for (int i = 0; i < total_layers_count - 1; i++)
            {
                n = layers[i].GetNeuronsCount();
                m = layers[i + 1].GetNeuronsCount();

                conns.Add(new ConnectionModel[n, m]);
                ConnectionModel[,] layerConnection = conns[i];
                for (int k = 0; k < n; k++)
                    for (int l = 0; l < m; l++)
                        layerConnection[k, l] = new ConnectionModel("conn" + i + "." + (i + 1) + "_" + k + l + "", TimeUnit.Sec, weights[ind++]); //2.0 * rand.NextDouble() - 1.0);
            }
        }
    }
}
