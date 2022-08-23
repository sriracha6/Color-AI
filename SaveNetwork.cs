using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace ActualColorAI
{
    public partial class Network 
    {
        /// <summary>
        /// Save the current state of the network to a file.
        /// </summary>
        public bool SaveState(string filepath)
        {
            List<string> o = new List<string>();
            string s = "";
            for(int i = 0; i < layers.Length; i++)
            {
                s += layers[i].numNodesIn + "|" + layers[i].numNodesOut + ( i == layers.Length - 1 ? "" : ",");
            }
            o.Add(s);
            foreach(Layer layer in layers)
            {
                o.Add("=");
                for(int nodeout = 0; nodeout < layer.numNodesOut; nodeout++)
                    for(int nodein = 0; nodein < layer.numNodesIn; nodein++)
                        o.Add(layer.weights[nodein, nodeout].ToString());
                o.Add("-");
                for(int nodeout = 0; nodeout < layer.numNodesOut; nodeout++)
                    o.Add(layer.biases[nodeout].ToString());
                o.Add("~");
                    for(int nodeout = 0; nodeout < layer.numNodesOut; nodeout++)
                        o.Add(layer.costGradientB[nodeout].ToString());
                o.Add("/");
                    for(int nodeout = 0; nodeout < layer.numNodesOut; nodeout++)
                        for(int nodein = 0; nodein < layer.numNodesIn; nodein++)
                            o.Add(layer.costGradientW[nodein, nodeout].ToString());
            }
            File.WriteAllLines(filepath, o.ToArray());  
            return true;
        }

        /// <summary>
        /// Load a network state.
        /// </summary>
        public static Network LoadState(string filepath)
        {
            string[] file = File.ReadAllLines(filepath);
            int i = 0;
            Network n = null;
            Layer cl = null;
            int l = 0;
            int mode = 0;
            double[,] curWeights = null;
            double[,] curCWeights = null;
            double[] curBiases = null;
            double[] curCBiases = null;
            int x,x2,y,y2,b,b2 = 0;
            x2=0; y2=0; x=0; y=0; b=0;
            foreach(string line in file)
            {
                if(i==0)
                {
                    n = new Network();
                    var q = line.Split(',');
                    n.layers = new Layer[q.Length];
                    for(int asd = 0; asd < q.Length; asd++)
                    {
                        string[] sizes = q[asd].Split('|');
                        n.layers[asd] = new Layer(int.Parse(sizes[0]), int.Parse(sizes[1]));
                    }
                    i++; continue; 
                }
                if(line=="=") {
                    if(cl!=null)
                    {
                        cl.weights = curWeights;
                        cl.biases = curBiases;
                        cl.costGradientB = curCBiases;
                        cl.costGradientW = curCWeights;
                    }
                    cl = n.layers[l];
                    mode = b = x = y = x2 = y2 = 0;
                    mode = 0; b = 0; x = 0; y = 0; x2 = 0; y2 = 0; b2 = 0;
                    l++;
                    curWeights = new double[cl.numNodesIn, cl.numNodesOut];
                    curBiases = new double[cl.numNodesOut];
                    curCBiases = new double[cl.numNodesOut];
                    curCWeights = new double[cl.numNodesIn, cl.numNodesOut];
                    i++; continue;
                }
                if(line == "-") 
                {
                    mode = 1;
                    x=y=0;
                    i++; continue;
                }
                if(line == "~")
                {
                    mode = 2;
                    i++; continue;
                }
                if(line == "/")
                {
                    mode = 3;
                    i++; continue;
                }
                if(mode == 0) // weights
                {
                    curWeights[x, y] = double.Parse(line);
                    x++;
                    if(x >= cl.numNodesIn)
                    {
                        x = 0;
                        y++;
                    }
                }
                else if (mode == 1) // biases
                {
                    curBiases[b] = double.Parse(line);
                    b++;
                }
                else if (mode == 2) // costGradientBiases
                {
                    curCBiases[b2] = double.Parse(line);
                    b2++;
                }
                else
                {
                    curCWeights[x2, y2] = double.Parse(line);
                    x2++;
                    if(x2 >= cl.numNodesIn)
                    {
                        x2 = 0;
                        y2++;
                    }
                }
                i++;
            }
            //catch { Console.WriteLine(i); }
            return n;
        }
    }
}