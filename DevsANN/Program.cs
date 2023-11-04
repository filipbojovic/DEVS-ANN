using DevsANN.Simulators;
using DEVSsharp;
using System;

namespace DevsANN
{
    class Program
    {
        static void Main(string[] args)
        {
            SRTEngine engine = new SRTEngine(new NeuralNetworkSimulator("ann"), 5.0, input);
            engine.RunConsoleMenu();
        }

        private static PortValue input(Devs model)
        {
            NeuralNetworkSimulator nn = (NeuralNetworkSimulator)model;

            if (nn != null)
            {
                if (Console.ReadLine().Trim().Equals("train"))
                    return new PortValue(nn.trainInputPort, null);
                else // testing phase
                    return new PortValue(nn.testInputPort, null);
            }

            return new PortValue(null, null);
        }
    }
}
