package rainer_sieberer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

public class CSVDigitConverter implements DataInterpreter
{

	public List<List<Double>> getTrainingData ( File trainingFile ) throws Exception
	{
		List<List<Double>> trainingData = new ArrayList<>();

		try (BufferedReader br = new BufferedReader(new FileReader( trainingFile )))
		{
			br.lines().forEach(s -> {
				String[] allValues = s.split(",", 2);

				// scale and shift the inputs
				trainingData.add(Arrays.stream(allValues[1].split(","))
						.mapToDouble(x -> Double.parseDouble(x) / 255.0 * 0.99 + 0.01)
						.boxed()
						.collect(Collectors.toList()));
			});
		}
		
		return trainingData;
	}

	public List<Integer> getTrainingDataSolution ( File trainingFile ) throws Exception
	{
		List<Integer> trainingDataSolution = new ArrayList<>();

		try (BufferedReader br = new BufferedReader(new FileReader( trainingFile )))
		{
			br.lines().forEach(s -> {
				String[] allValues = s.split(",", 2);
				trainingDataSolution.add(Integer.parseInt(allValues[0]));
			});
		}

		return trainingDataSolution;
	}

	public double[] transformInput ( String inputString ) throws Exception
	{
		return Arrays.stream( inputString.split(",") )
				.mapToDouble(s -> Double.parseDouble(s) / 255.0 * 0.99 + 0.01)
				.toArray();
	}

	private double[] toArray(Matrix matrix)
	{
		double[] result = new double[matrix.getRowDimension()];

		for (int i = 0; i < result.length; i++)
			result[i] = matrix.get(i, 0);

		return result;
	}

}
