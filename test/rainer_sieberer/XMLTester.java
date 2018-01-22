package rainer_sieberer;

import java.io.File;

public class XMLTester
{

	public static void main( String[] args)
	{
		try{
		XMLReader xmlreader0 = new XMLReader();
		NeuralNetwork net = xmlreader0.buildNNfromXMLFile(new File("../data/netExample.xml"));
		net.printInfo();
		} catch ( Exception e) { e.printStackTrace(); }
	}

}
