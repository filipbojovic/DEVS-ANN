using DevsANN.Data;
using DEVSsharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace DevsANN.Models
{
    class NeuronModel : Atomic
    {
        readonly double eps_z;
        readonly double tau_delay;
        readonly int outputsCount; // number of outputs from this neuron
        int waitCounter; // wait for waitCounter number of inputs before propagating output value further
        double previousSignal; // the last input to neuron Z_j
        double lastOutput; // the last output propagated further (when an activation function is applied to the input) A_j

        public List<InputPort> inputPorts; // it is used to send data to the neuron inputs
        public OutputPort outputPort; // it is used to send the output value further
        public OutputPort deltaErrorOutputPort; // the change of error C with a change of the input 'input' (dC / d_input)
        public InputPort startBackPropInputPort; // it is used only for the neurons in the input layer
        public InputPort deltaErrorInputPort; // the change of error C with a change of input for a neuron in the next layer (L + 1)

        private List<double> deltaInputErrors; // it is used to store dC/d_input values of the next layer (L + 1)
        private List<double> inputValues;
        public SortedList<double, double> taus;

        // ----------- NOVO ---------------
        List<double> errorDerivativesValues;
        List<double> totalErrorDerivativesValues;
        List<double> previousSignalsList;
        int deltaErrorListCounter = 0;
        // --------------------------------

        Queue<PortValue> msgBuff;
        public NeuronModel(string name, TimeUnit tu, int inputsCount, double eps_z, double tau_delay, int outputsCount) : base(name, tu)
        {
            this.eps_z = eps_z;
            this.outputsCount = outputsCount;
            this.tau_delay = tau_delay;

            deltaErrorOutputPort = AddOP(name + "deltaErrorOutputPort");
            deltaErrorInputPort = AddIP(name + "deltaErrorInputPort");
            startBackPropInputPort = AddIP(name + "startBackPropInputPort");
            outputPort = AddOP(name + "outputPort");
            inputPorts = new List<InputPort>();
            for (int i = 0; i < inputsCount; i++)
                inputPorts.Add(AddIP(name + "inputPort" + i + ""));

            init();
        }

        public override bool delta_x(PortValue x)
        {
            for (int i = 0; i < inputPorts.Count; i++)
                if (x.port == inputPorts[i])
                {
                    inputValues[i] = (double)x.value;
                    ++waitCounter;
                    double sum = inputValues.Sum();

                    if (previousSignal == -1.0 || Math.Abs(previousSignal - sum) > eps_z)
                        previousSignal = sum;

                    if (waitCounter == inputPorts.Count) // if it is a neuron in the output layer, calculate the error
                    {
                        waitCounter = 0;
                        double triggerTime = TimeCurrent + tau_delay;

                        previousSignalsList.Add(previousSignal);

                        if (taus.ContainsKey(triggerTime))
                            taus[triggerTime] = previousSignal;
                        else
                            taus.Add(triggerTime, previousSignal);

                        return true; // call tau()
                    }
                    return false;
                }

            if (x.port == startBackPropInputPort) // for the neurons in the output layer
            {
                errorDerivativesValues = (List<double>)x.value; // list dc/dout

                for (int i = 0; i < errorDerivativesValues.Count; i++)
                    errorDerivativesValues[i] *= Config.primActivationFunction(previousSignalsList[i]);

                msgBuff.Enqueue(new PortValue(deltaErrorOutputPort, errorDerivativesValues));

                return true;
            }
            else if (x.port == deltaErrorInputPort) // this port is hit by ConnectionModel
            {
                errorDerivativesValues = (List<double>)x.value; // the last value in this list is always connection weight
                double connectionWeight = errorDerivativesValues[errorDerivativesValues.Count - 1];

                if (deltaErrorListCounter == 0)
                {
                    totalErrorDerivativesValues.Clear();
                    for (int i = 0; i < errorDerivativesValues.Count - 1; i++)
                        totalErrorDerivativesValues.Add(0.0);
                }

                for (int i = 0; i < errorDerivativesValues.Count - 1; i++) // calculated a part of delta error based on the list sent from a neuron in the next layer (L + 1)
                    totalErrorDerivativesValues[i] += (connectionWeight * errorDerivativesValues[i] * Config.primActivationFunction(previousSignalsList[i]));
                deltaErrorListCounter++;

                if (deltaErrorListCounter == outputsCount) // all derivatives dC/d_input(L+1) are arrived
                {
                    msgBuff.Enqueue(new PortValue(deltaErrorOutputPort, totalErrorDerivativesValues));
                    deltaErrorListCounter = 0;
                    return true;
                }
                return false;
            }

            return false;
        }
        public override void delta_y(ref PortValue y)
        {
            if (msgBuff.Count > 0)
            {
                y = msgBuff.Dequeue();
                return;
            }

            lastOutput = Config.activationFunction(taus.Values[0]);
            y.Set(outputPort, lastOutput);
            taus.RemoveAt(0);
            return;
        }
        public override void init()
        {
            previousSignal = -1.0;
            waitCounter = 0;

            deltaInputErrors = new List<double>();
            msgBuff = new Queue<PortValue>();
            inputValues = new List<double>();
            for (int i = 0; i < inputPorts.Count; i++)
                inputValues.Add(0.0);

            taus = new SortedList<double, double>();
            totalErrorDerivativesValues = new List<double>();

            previousSignalsList = new List<double>();
        }
        public override double tau()
        {
            if (taus.Count > 0)
                return taus.Keys[0] - TimeCurrent;

            if (msgBuff.Count > 0)
                return 0.0;

            return double.PositiveInfinity;
        }
    }
}
