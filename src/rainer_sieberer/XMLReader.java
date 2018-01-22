package rainer_sieberer;

import java.io.File;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import org.w3c.dom.*;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.lang.RuntimeException;

/**
 * @author	Jonas Sieberer
 * @author	Rudolf Rainer
 */
public class XMLReader
{

	/**
	 * This method takes a File (which is expected to be an XML file) and makes a list of
	 * items out the information in this file.
	 *
	 * @param		filename	the method tries to transform the tree structure of the xml file
	 *										into an ItemList
	 * @return						the ItemList containing the elements described in the file 'filename'
	 */
	public static NeuralNetwork buildNNfromXMLFile ( File filename ) throws Exception
	{ 
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		DocumentBuilder db = dbf.newDocumentBuilder();
		Document doc = db.parse(filename);
		Element root = doc.getDocumentElement();

		int epochs;
		double learningRate;
		ArrayList<Integer> numberOfNodesList = new ArrayList<Integer>();
		List<Function<Double, Double>> activation = new ArrayList<Function<Double, Double>>();
		ArrayList<String> initialValues = new ArrayList<String>();
		File trainingFile;

		if ( root.getAttribute("epochs").equals("") )
			epochs = 5;
		else epochs = Integer.parseInt( root.getAttribute("epochs") );

		if ( root.getAttribute("learningRate").equals("") )
			learningRate = 0.1;
		else learningRate = Double.parseDouble( root.getAttribute("learningRate") );

		if ( root.getAttribute("inputNodes").equals("") )
			throw new RuntimeException("size of input not declared in XML file!");
		else numberOfNodesList.add( Integer.parseInt( root.getAttribute("inputNodes") ) );

		if ( root.getAttribute("trainingFile").equals("") )
			throw new RuntimeException("no training file declared in XML file!");
		else trainingFile = new File( root.getAttribute("trainingFile") );

		NodeList listOfNodes = root.getChildNodes();

		for ( int i = 0; i < listOfNodes.getLength(); i++ )
		{			
			Node currentNode = listOfNodes.item(i);
			if ( currentNode.getNodeType() == Node.ELEMENT_NODE )
			{
				Element currentElement = (Element) currentNode;

				if ( currentElement.getAttribute("nodes").equals("") )
					throw new RuntimeException("number of nodes of a layer not declared in XML file!");
				else numberOfNodesList.add( Integer.parseInt( currentElement.getAttribute("nodes") ) );

				if ( currentElement.getAttribute("activationFunction").equals("") )
					activation.add( (Double x) -> 1.0 / (1.0 + Math.exp(-x)) );
				else if ( currentElement.getAttribute("activationFunction").equals("Sigmoid") )
					activation.add( (Double x) -> 1.0 / (1.0 + Math.exp(-x)) );
				// here other activation functions could be declared

				initialValues.add( currentElement.getAttribute("initialValues") );
			}
		}

		int[] numberOfNodes = new int[numberOfNodesList.size()];
		for ( int i = 0; i < numberOfNodesList.size(); i++ )
		{
			numberOfNodes[i] = numberOfNodesList.get(i);
		}

		NeuralNetwork net = new NeuralNetwork();
		net.init( trainingFile, numberOfNodes, epochs, learningRate, activation );
		
		for ( int i = 0; i < initialValues.size(); i++ )
		{
			if ( initialValues.get(i).equals("zero") )
				net.changeInitialValues( i, new Matrix ( numberOfNodes[i], numberOfNodes[i+1] ) );
		}

		return net;
	}

/*
	private static void addListToList ( Element e, ItemList list )
	{
		NodeList listOfNodes = e.getChildNodes(); // extract the 'subitems'

		for (int i=0;i<listOfNodes.getLength();i++){

			Node currentNode = listOfNodes.item(i);
			int type = currentNode.getNodeType();

			if (type == Node.ELEMENT_NODE){

				Element currentElement = (Element) currentNode;
				if ( currentElement.getTagName().equals("list") )
					addListToList ( currentElement, list ); // if it is a list, a recursion is triggered
				else 
					addElementToList ( currentElement, list ); // is it is a single element, use the other method.

			}
		}
	} 

	private static void addElementToList ( Element currentElement, ItemList list )
	{
		if ( currentElement.getTagName().equals("book") )
		{
			Item newItem = new Book(currentElement.getAttribute("name"),
				Double.parseDouble(currentElement.getAttribute("price")),
				Integer.parseInt(currentElement.getAttribute("isbn")));
			list.add(newItem);
		} else
		if ( currentElement.getTagName().equals("cd") )
		{
			Item newItem = new Cd(currentElement.getAttribute("name"),
				Double.parseDouble(currentElement.getAttribute("price")));
			list.add(newItem);
		} else
		{
			Item newItem = new Item(currentElement.getAttribute("name"),
				Double.parseDouble(currentElement.getAttribute("price")));
			list.add(newItem);
		} 
	} */

}
