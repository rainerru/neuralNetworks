package rainer_sieberer;

import java.io.File;

/**
 * 
 * (initial python implementation: Make Your Own Neural Network, Tariq Rashid)
 * (this code is based on submissions from C.Moesl, A.Schuetz, T.Hilgart, M.Regirt)
 */
public class DigitRecognizer 
{

	MachineLearner ml;

	public DigitRecognizer ( File data, File config )
	{
		CSVDigitConverter interpreter = new CSVDigitConverter();
		ml = new MachineLearner( data, config, interpreter );
	}

	public int recognize ( String inputString ) throws Exception
	{
		return ml.recognize( inputString );
	}
	
}
