using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

// This is a NEural Network. It is a type of AI.
// It simulates the human brain. Next time I'm doing NEAT. This is overly complicated.
// (it actually wasn't that bad)
// it wasn't that bad to make but it's that bad when it gives its outputs.

// anyways, it takes in a color as an input and then it tells if you should use black or white text.
// output 1 is black, output 2 is white.
// so if the output is 1,0 then it's black
// if its 0,1 then it's white
// if its neither then something went drastically wrong
namespace ActualColorAI
{
	internal class Program
	{
		public static void Main(string[] args)
		{
			Network n = null;
			if(Directory.GetFiles("./Data").Length == 0)
				GenerateData();

			DataPoint[] totalData = new DataPoint[Directory.GetFiles("./Data").Length].OrderBy(x=>Guid.NewGuid()).ToArray();

			int i = 0;
			foreach(string file in Directory.GetFiles("./Data"))
			{
				string[] lines = File.ReadAllLines(file);
				totalData[i] = ParseData(file);
				i++;
			}
			Console.WriteLine("Read from local STATE.brain file? (True/False)");
			bool sexy = bool.Parse(Console.ReadLine()); // im very lazy. dont even wanna add a utility function im that lazy
			if(sexy) { n = Network.LoadState("STATE.brain"); goto end; }
			Console.WriteLine("Learn Rate Percent (Recommended - 20): ");
			int learnRate = int.Parse(Console.ReadLine());
			Console.WriteLine("Sample Size Percent (Recommended - 80): ");
			float sampleSize = float.Parse(Console.ReadLine()) / 100f;
			Console.WriteLine("Activation Function (Recommended - Sigmoid) OPTIONS: Sigmoid, Step, Relu: ");
			Layer.activation = Console.ReadLine();
			Console.WriteLine("Hidden Layer Sizes (Comma seperated) (Recommended - 5):");
			List<string> hlsizestr = Console.ReadLine().Split(',').ToList();
			List<int> hlsize = new List<int>();
			hlsize.Add(3); // INPUT
			foreach(string s in hlsizestr) hlsize.Add(int.Parse(s)); 
			hlsize.Add(2); // OUTPUT
			Console.WriteLine("Learning Iterations (Recommended - 10000+): ");
			int iterations = int.Parse(Console.ReadLine());
			Console.WriteLine("Use Backpropagation (Recommended - True): ");
			bool isBackProp = bool.Parse(Console.ReadLine());
			n = new Network(hlsize.ToArray());
			//Console.WriteLine("Learn Iterations: ");
			//int iterations = int.Parse(Console.ReadLine());

			DataPoint[] accesableData = totalData.ToList().Take((int)(sampleSize * totalData.Length)).ToArray();
			DataPoint[] testData = totalData.ToList().Skip((int)(sampleSize * totalData.Length)).ToArray();

			for(int it = 0; it < iterations; it++)
			{
				DataPoint[] miniBatch = accesableData.ToList().OrderBy(x=>Guid.NewGuid()).Take(accesableData.Length / 10).ToArray();
				if(!isBackProp)
					n.Learn(miniBatch, learnRate / 100D);  // we need to only give it access to some of the data so the rest is used for testing   
				else
					n.LearnFAST(miniBatch, learnRate / 100D);
			}
			
			Console.WriteLine("===========");
			foreach(DataPoint d in testData)
			{
				int answer = n.Answer(d.inputs);
				var hColor = System.Drawing.Color.FromArgb((int)(d.inputs[0]*255), (int)(d.inputs[1]*255), (int)(d.inputs[2]*255));
				var s = (string)(" Hello ".Highlight(hColor));
				Console.WriteLine(s.Color(answer == 0 ? System.Drawing.Color.Black : System.Drawing.Color.White));
				Console.WriteLine("Confidence: " + (int)(n.CalcOutputs(d.inputs)[answer] * 100) + "%");
				Console.WriteLine("Expected: " + (d.expectedOutputs[0] == 1 ? "Black" : "White"));
				Console.WriteLine("===========");
			}
			Console.WriteLine("COST: ".Color(System.Drawing.Color.Red) + n.Cost(testData) + " (Lower values = Better accuracy)");
			Console.WriteLine("===========");
			n.SaveState("./STATE.brain");
			end:
			for(;;)
			{
				Console.WriteLine("Enter a color input (r,g,b):");
				string inp = Console.ReadLine();
				string[] inputs = inp.Split(',');
				var data = new double[] { (int.Parse(inputs[0]))/255D, (int.Parse(inputs[1]))/255D, (int.Parse(inputs[2]))/255D };

				int answer = n.Answer(data);
				var hColor = System.Drawing.Color.FromArgb(int.Parse(inputs[0]), int.Parse(inputs[1]), int.Parse(inputs[2]));
				var s = (string)(" Hello ".Highlight(hColor));
				Console.WriteLine(s.Color(answer == 0 ? System.Drawing.Color.Black : System.Drawing.Color.White));
				Console.WriteLine("Confidence: " + (int)(n.CalcOutputs(data)[answer] * 100) + "%");
				Console.WriteLine("===========");
			}
		}

		public static DataPoint ParseData(string path)
		{
			string[] file = File.ReadAllLines(path);
			// this is divided by 255 to normalize it. You always want normalized values in an AI
			return new DataPoint(new double[] {double.Parse(file[0])/255D, double.Parse(file[1])/255D, double.Parse(file[2])/255D},
						new double[] {double.Parse(file[3]), double.Parse(file[4])});
		}

		public static void GenerateData()
		{
			Random r = new Random();
			for(int i = 0; i < 100; i++)
			{
				string path = "./Data/" + i + ".trainingdata";
				var da = new string[] {r.Next(0, 255).ToString(), r.Next(0, 255).ToString(), r.Next(0, 255).ToString()};
				File.WriteAllLines(path, da);
				Console.WriteLine(i + " " + " ".Highlight(System.Drawing.Color.FromArgb(int.Parse(da[0]), int.Parse(da[1]), int.Parse(da[2]))));
			}
			Console.WriteLine("Now please, go add the expected outputs.");
		}
	}
}