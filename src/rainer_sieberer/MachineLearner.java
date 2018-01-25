package rainer_sieberer;

import java.io.File;

/**
 *	this is the framework
 */
public class MachineLearner
{

	private NeuralNetwork net;
	private DataInterpreter interpreter;

	public MachineLearner ( File data, File config, DataInterpreter interpreter )
	{
		this.interpreter = interpreter;
		try
		{
			net = XMLReader.buildNNfromXMLFile( config );
			net.train(
				this.interpreter.getTrainingData( net.getTrainingFile() ),
				this.interpreter.getTrainingDataSolution( net.getTrainingFile() )
			);
		}	catch ( Exception e ) { e.printStackTrace(); }
	}

	public int recognize ( String inputString ) throws Exception
	{
		return indexFromMax( net.query( this.interpreter.transformInput( inputString ) ) );
	}

	private int indexFromMax(double[] data)
	{
		int max = 0;

		for (int i = 0; i < data.length; i++)
			if (data[i] > data[max])
				max = i;

		return max;
	}

}
