using System;
using System.Linq;
using System.Collections.Generic;

namespace ActualColorAI
{
    public partial class Network
    {
        Layer[] layers;

        public Network(params int[] layerSizes)
        {
            layers = new Layer[layerSizes.Length - 1];
            for(int i = 0; i < layers.Length; i++)
                layers[i] = new Layer(layerSizes[i], layerSizes[i+1]);
        }
        public Network(){}

        public double[] CalcOutputs(double[] inputs)
        {
            foreach(Layer layer in layers)
                inputs = layer.CalculateOutputs(inputs);
            return inputs;
        }

        public int Answer(double[] inputs)
        {
            double[] outputs = CalcOutputs(inputs);
            return BiggestOutput(outputs);
        }

        public double Cost(DataPoint dataPoint)
        {
            double[] outputs = CalcOutputs(dataPoint.inputs);
            Layer outputLayer = layers[layers.Length - 1];
            double cost = 0;

            for(int nodeout = 0; nodeout < outputs.Length; nodeout++)
                cost += outputLayer.NodeCost(outputs[nodeout], dataPoint.expectedOutputs[nodeout]);
            return cost;
        }

        public double Cost(DataPoint[] data)
        {
            double cost = 0;
            foreach(DataPoint d in data)
                cost += Cost(d);
            return cost / data.Length;
        }

        public int BiggestOutput(double[] outputs)
        {
            int record = 0;
            for(int i =0; i < outputs.Length; i++)
                if(outputs[i] > outputs[record])
                    record = i;
            return record;
        }

        /*public void Learn(double learnRate)
        {
            const double h = 0.00001;
           
            double delta = Funct(inputValue + h) - Funct(inputValue);
            double slope = delta / h;

            inputValue -= slope * learnRate;
        }*/

        public void Learn(DataPoint[] trainingData, double learningRate)
        {
            const double h = 0.0001;
            double originalCost = Cost(trainingData);
            foreach(Layer layer in layers)
            {
                for(int nodein = 0; nodein < layer.numNodesIn; nodein++)
                {
                    for(int nodeout = 0; nodeout < layer.numNodesOut; nodeout++)
                    {
                        layer.weights[nodein, nodeout] += h;
                        double delta = Cost(trainingData) - originalCost;
                        layer.weights[nodein, nodeout] -= h;
                        layer.costGradientW[nodein, nodeout] = delta/h;
                    }
                }
                for(int biasi = 0; biasi < layer.biases.Length; biasi++)
                {
                    layer.biases[biasi] += h;
                    double delta = Cost(trainingData) - originalCost;
                    layer.biases[biasi] -= h;
                    layer.costGradientB[biasi] = delta / h;
                }
            }
            ApplyAllGradients(learningRate); // call applygradient() on all layers
        }
        public void ApplyAllGradients(double learningRate)
        {
            foreach(Layer layer in layers)
                layer.ApplyGradients(learningRate);
        }

        public void LearnFAST(DataPoint[] batch, double learnRate)
        {
            // backpropgated
            foreach(DataPoint d in batch)
            {
                UpdateAllGradients(d);
            }

            ApplyAllGradients(learnRate / batch.Length);
            ZeroGradients();
        }

        public void ZeroGradients()
        {
            foreach(Layer l in layers)
            {
                l.ZeroGradients();
            }
        }

        public void UpdateAllGradients(DataPoint d)
        {
            CalcOutputs(d.inputs);
            Layer output = layers[layers.Length - 1];
            double[] nodeValues = output.CalcOutLayerNodes(d.expectedOutputs);
            output.UpdateGradients(nodeValues);
            for(int i = layers.Length - 2; i >= 0; i--)
            {
                nodeValues = layers[i].CalcHiddenLayerValues(layers[i + 1], nodeValues);
                layers[i].UpdateGradients(nodeValues);
            }
        }
    }
}