using System;

namespace ActualColorAI
{
    public class DataPoint
    {
        public double[] inputs;
        public double[] expectedOutputs;

        public DataPoint(double[] inputs, double[] expectedOutputs)
        {
            this.inputs = inputs;
            this.expectedOutputs = expectedOutputs;
        }
    }
}