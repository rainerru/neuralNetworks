package rainer_sieberer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.List;
import java.util.Random;
import java.util.Collections;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class DigitTester
{

	public static void main ( String[] args )
	{
		DigitRecognizer recognizer = new DigitRecognizer(
			new File("../data/mnist_train_100.csv"),
			new File("../data/netExample.xml")
		);

		int attemptsOk = 0;
		int attemptsFailed = 0;
		try (Stream<String> stream = Files
				.lines(Paths.get("../data/mnist_test_10.csv"))) {
			List<String> testDataList = stream.collect(Collectors.toList());

			System.out.println("correct | recognized");

			// go through all the records in the test data set
			for (String record : testDataList) {
				// split the record by the ',' commas
				String[] allValues = record.split(",", 2);

				// correct answer is first value
				int correctDigit = Integer.parseInt(allValues[0]);
				int recognizedDigit = recognizer.recognize(allValues[1]);

				if (correctDigit == recognizedDigit)
					attemptsOk++;
				else
					attemptsFailed++;

				System.out.println("  " + correctDigit + "     |     " + recognizedDigit);
			}
		} catch ( Exception e ) { e.printStackTrace(); }
		// calculate the performance score, the fraction of correct answers
		System.out.println("performance = " + (double) attemptsOk / (attemptsOk + attemptsFailed) * 100 + "%");
	}

}
