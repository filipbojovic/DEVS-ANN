using DevsANN.Data;
using DEVSsharp;
using System.Collections.Generic;
using System.Linq;

namespace DevsANN.Models
{
    class ConnectionModel : Atomic
    {
        private double weight;
        private double lastInput; // the last input value to this connection weight
        private int readedBatches;
        private List<double> deltaWeights;
        private Queue<PortValue> msgBuff;

        public InputPort inputNeuronPort;
        public OutputPort outputNeuronPort;
        public InputPort backPropInputPort;
        public OutputPort backPropOutputPort;


        List<double> deltaErrors;
        List<double> lastInputs;
        List<double> dws;

        public ConnectionModel(string name, TimeUnit tu, double weight) : base(name, tu)
        {
            this.weight = weight;

            inputNeuronPort = AddIP(name + "inputNeuronPort");
            outputNeuronPort = AddOP(name + "outputNeuronPort");
            backPropInputPort = AddIP(name + "backPropInputPort");
            backPropOutputPort = AddOP(name + "backPropOutputPort");

            init();
        }

        public override bool delta_x(PortValue x)
        {
            if (x.port == inputNeuronPort)
            {
                lastInput = (double)x.value;
                lastInputs.Add(lastInput);
                msgBuff.Enqueue(new PortValue(outputNeuronPort, lastInput * weight));
                return true;
            }
            else if (x.port == backPropInputPort)
            {
                deltaErrors.Clear();
                List<double> temp = (List<double>)x.value; // list of derivatives dC_dt

                for (int i = 0; i < temp.Count; i++)
                {
                    dws.Add(0.0);
                    dws[i] = temp[i] * lastInputs[i];
                }

                double dw = dws.Sum() / Config.BATCH_SIZE;

                for (int i = 0; i < temp.Count; i++)
                    deltaErrors.Add(temp[i]);
                deltaErrors.Add(weight);

                weight -= Config.LEARNING_RATE * dw;

                msgBuff.Enqueue(new PortValue(backPropOutputPort, deltaErrors)); // propagathe further

                dws.Clear();

                return true;
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
            deltaWeights = new List<double>();
            deltaErrors = new List<double>();
            lastInputs = new List<double>();
            dws = new List<double>();
            readedBatches = 0;
        }

        public override double tau()
        {
            if (msgBuff.Count > 0)
                return 0.0;

            return double.PositiveInfinity;
        }

        public override string Get_s()
        {
            return weight.ToString();
        }
    }
}
