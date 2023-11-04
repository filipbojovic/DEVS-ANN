﻿using DEVSsharp;
using System.Collections.Generic;

namespace DevsANN.Models
{
    class LayerModel : Coupled
    {
        public List<NeuronModel> neurons;
        public LayerModel(string name, int neurons_count, int neuron_inputs, double eps_z, double tau_delay, int neuron_outputs) : base(name)
        {
            neurons = new List<NeuronModel>();

            for (int i = 0; i < neurons_count; i++)
                neurons.Add(new NeuronModel(name + "neuron" + i + "", TimeUnit.Sec, neuron_inputs, eps_z, tau_delay, neuron_outputs));
        }

        public int GetNeuronsCount()
        {
            return neurons.Count;
        }
    }
}
