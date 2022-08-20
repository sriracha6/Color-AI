using System;

namespace ActualColorAI
{
    public class Layer
    {
        public int numNodesIn;
        public int numNodesOut;
        public double[,] weights;
        public double[,] costGradientW;
        public double[] costGradientB;
        public double[] biases;

        public static string activation = "sigmoid";
        
        public Layer(int numin, int numout)
        {
            this.numNodesIn = numin;
            this.numNodesOut = numout;
            
            costGradientW = new double[numin, numout];
            costGradientB = new double[numout];
            weights = new double[numin, numout];
            biases = new double[numout]; 
            InitRandomWeights();
        }
        
        public double[] CalculateOutputs(double[] inputs)
        {
            double[] weightedInputs = new double[numNodesOut];
            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
            {
                double weightedIn = biases[nodeOut];
                for(int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
                {
                    weightedIn += inputs[nodeIn] * weights[nodeIn, nodeOut];
                }
                weightedInputs[nodeOut] = Activation(weightedIn);
            }
            return weightedInputs;
        }

        public double NodeCost(double outputActivation, double expectedOut)
        {
            double error = outputActivation - expectedOut; // this is what cost is. smaller == better 
            return error * error; // emphasize larger error amounts.
        }

        public double Activation(double input)
        {
            if(activation.ToLower() == "sigmoid")
                return 1/(1 + Math.Exp(-input));
            else if (activation.ToLower() == "relu")
                return input > 0 ? input : 0;
            else if (activation.ToLower() == "step")
                return input > 0 ? 1 : 0;
            else return input;
        }

        public void ApplyGradients(double learnrate)
        {
            for(int nodeout = 0; nodeout < numNodesOut; nodeout++)
            {
                biases[nodeout] -= costGradientB[nodeout] * learnrate;
                for(int nodein = 0; nodein < numNodesIn; nodein++)
                {
                    weights[nodein, nodeout] -= costGradientW[nodein, nodeout] * learnrate;
                }
            }
        }

        public void InitRandomWeights()
        {
            System.Random r = new System.Random();
            for(int nodeout = 0; nodeout < numNodesOut; nodeout++)
            {
                for(int nodein = 0; nodein < numNodesIn; nodein++)
                {
                    weights[nodein, nodeout] = (r.NextDouble() * 2 - 1) / Math.Sqrt(numNodesIn);
                }
            }
        }
    }
}