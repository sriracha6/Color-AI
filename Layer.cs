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
        public double[] inputs;
        public double[] activations;
        public double[] weightedInputs;

        public static string activation = "sigmoid";
        
        public Layer(int numin, int numout, bool randomWeights=true)
        {
            this.numNodesIn = numin;
            this.numNodesOut = numout;
            
            this.costGradientW = new double[numin, numout];
            this.costGradientB = new double[numout];
            this.weights = new double[numin, numout];
            this.biases = new double[numout]; 
            this.weightedInputs = new double[numout];
            if(randomWeights) InitRandomWeights();
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
                this.weightedInputs[nodeOut] = weightedIn;
                weightedInputs[nodeOut] = Activation(weightedIn);
            }
            this.inputs = inputs;
            this.activations = weightedInputs;
            return weightedInputs;
        }

        public double NodeCost(double outputActivation, double expectedOut)
        {
            double error = outputActivation - expectedOut; // this is what cost is. smaller == better 
            return error * error; // emphasize larger error amounts.
            // BUT NO!! you can find the deriviatie of this to make it faster:
            // return 2 * (outputActivation - expectedOut);
        }

        public double CostDeriv(double outputActivation, double expectedOut)
        {
            return 2 * (outputActivation - expectedOut);
        }

        public double Activation(double input)
        {
            if(activation.ToLower() == "sigmoid")
            {
                // DERIVATIVED!
                //double s = 1/(1 + Math.Exp(-input));
                //return s * (1 - s);
                return 1/(1 + Math.Exp(-input));
            }
            else if (activation.ToLower() == "relu")
                return input > 0 ? input : 0;
            else if (activation.ToLower() == "step")
                return input > 0 ? 1 : 0;
            else return input;
        }

        public double SigmoidDeriv(double input)
        {
            double s = 1/(1 + Math.Exp(-input));
            return s * (1 - s);
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

        public double[] CalcOutLayerNodes(double[] expected)
        {
            double[] nodeValues = new double[expected.Length];
            for(int i = 0; i < nodeValues.Length; i++)
            {
                double costDeriv = CostDeriv(activations[i], expected[i]);
                double activDeriv = SigmoidDeriv(weightedInputs[i]);
                nodeValues[i] = activDeriv * costDeriv;
            }
            return nodeValues;
        }

        public void UpdateGradients(double[] nodevalues)
        {
            for(int nodeout = 0; nodeout < numNodesOut; nodeout++)
            {
                for(int nodein = 0; nodein < numNodesIn; nodein++)
                {
                    double costDeriv = inputs[nodein] * nodevalues[nodeout];
                    costGradientW[nodein, nodeout] += costDeriv;
                }
                double derivCostWeightBias = nodevalues[nodeout];
                costGradientB[nodeout] += derivCostWeightBias;
            }
        }

        public double[] CalcHiddenLayerValues(Layer oldLayer, double[] oldValues)
        {
            double[] newNodes = new double[numNodesOut];

            for(int newnodei = 0; newnodei < newNodes.Length; newnodei++)
            {
                double newvalue = 0;
                for(int oldnodei = 0; oldnodei < oldValues.Length; oldnodei++)
                {
                    double deriv = oldLayer.weights[newnodei, oldnodei];
                    newvalue += deriv * oldValues[oldnodei];
                }
                newvalue += SigmoidDeriv(weightedInputs[newnodei]);
                newNodes[newnodei] = newvalue;
            }

            return newNodes;
        }

        public void ZeroGradients()
        {
            costGradientB = new double[numNodesOut];
            costGradientW = new double[numNodesIn, numNodesOut];
        }
    }
}