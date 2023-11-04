namespace DevsANN.Models
{
    class FlowerModel
    {
        public double SepalLength { get; set; }
        public double SepalWidth { get; set; }
        public double PetalLength { get; set; }
        public double PetalWidth { get; set; }
        public int FlowerType { get; set; }

        public double GetDataByNeuronID(int index)
        {
            switch (index)
            {
                case 0:
                    return SepalLength;
                case 1:
                    return SepalWidth;
                case 2:
                    return PetalLength;
                case 3:
                    return PetalWidth;
                default:
                    return 0.0;
            }
        }
    }
}
