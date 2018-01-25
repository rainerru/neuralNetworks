package rainer_sieberer;

import java.io.File;
import java.util.List;

public interface DataInterpreter
{

	public List<List<Double>> getTrainingData ( File trainingFile ) throws Exception;

	public List<Integer> getTrainingDataSolution ( File trainingFile )  throws Exception;

	public double[] transformInput ( String inputString ) throws Exception;

}
