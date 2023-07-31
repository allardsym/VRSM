using Microsoft.ML.Data;
using Microsoft.ML;
using System.Data;

while (true)
{
	ModelInput modelInput = new ModelInput();
	
	Console.WriteLine("Position");
	modelInput.Position = Console.ReadLine();
	Console.WriteLine("Hands");
	modelInput.Hands = Int32.Parse(Console.ReadLine());
	Console.WriteLine("Bodyparts");
	modelInput.Bodyparts = Console.ReadLine();
	Console.WriteLine("Injuries");
	modelInput.Injuries = Console.ReadLine();
	Console.WriteLine("Symptoms");
	modelInput.Symptoms = Console.ReadLine();
	Console.WriteLine("Goals");
	modelInput.Goals = Console.ReadLine();

	var mlContext = new MLContext();
	var transformer = mlContext.Model.Load(Path.GetFullPath("MLModel3.zip"), out _);
	var predictionEngine = mlContext.Model
		.CreatePredictionEngine<ModelInput, ModelOutput>(transformer);
	var modelOutput = predictionEngine.Predict(modelInput);
	var labelBuffer = new VBuffer<ReadOnlyMemory<char>>();
	predictionEngine.OutputSchema["Score"].Annotations.GetValue("SlotNames", ref labelBuffer);
	var labels = labelBuffer.DenseValues().Select(l => l.ToString()).ToArray();

	var topScores = labels.ToDictionary(l => l, l => (decimal)modelOutput.Score[Array.IndexOf(labels, l)])
		.OrderByDescending(kv => kv.Value)
		.Take(3);

	foreach (var x in topScores)
	{
		Console.WriteLine(x.Key + " " + x.Value);
	}

	Console.WriteLine();
	Console.WriteLine();
	Console.WriteLine();
	Console.WriteLine();
}

public class ModelInput
{
	[ColumnName(@"Position")]
	public string Position { get; set; }

	[ColumnName(@"Hands")]
	public float Hands { get; set; }

	[ColumnName(@"Bodyparts")]
	public string Bodyparts { get; set; }

	[ColumnName(@"Injuries")]
	public string Injuries { get; set; }

	[ColumnName(@"Symptoms")]
	public string Symptoms { get; set; }

	[ColumnName(@"Goals")]
	public string Goals { get; set; }

	[ColumnName(@"Outcome")]
	public string Outcome { get; set; }

}

public class ModelOutput
{
	[ColumnName(@"Position")]
	public float[] Position { get; set; }

	[ColumnName(@"Hands")]
	public float Hands { get; set; }

	[ColumnName(@"Bodyparts")]
	public float[] Bodyparts { get; set; }

	[ColumnName(@"Injuries")]
	public float[] Injuries { get; set; }

	[ColumnName(@"Symptoms")]
	public float[] Symptoms { get; set; }

	[ColumnName(@"Goals")]
	public float[] Goals { get; set; }

	[ColumnName(@"Outcome")]
	public uint Outcome { get; set; }

	[ColumnName(@"Features")]
	public float[] Features { get; set; }

	[ColumnName(@"PredictedLabel")]
	public string PredictedLabel { get; set; }

	[ColumnName(@"Score")]
	public float[] Score { get; set; }

}