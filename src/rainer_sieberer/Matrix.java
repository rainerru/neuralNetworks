package rainer_sieberer;

import java.lang.StringBuilder;
import java.util.function.Function;

/**
 * sample code for PS Software Engineering, ws17: starting point for A12
 * (initial python implementation: Make Your Own Neural Network, Tariq Rashid)
 * (this code is based on submissions from C.Moesl, A.Schuetz, T.Hilgart, M.Regirt)
 */
public class Matrix {

	private final int row;
	private final int col;

	private double[][] elements;

	public Matrix(int row, int col) {
		this.row = row;
		this.col = col;
		this.elements = new double[row][col];
	}

	public int getRowDimension() {
		return row;
	}

	public int getColumnDimension() {
		return col;
	}

	public Matrix transposeMatrix() {
		Matrix B = new Matrix(this.col, this.row);
		for (int row = 0; row < this.col; row++)
			for (int col = 0; col < this.row; col++)
				B.set(row, col, this.get(col, row));
		return B;
	}

	public void set(int row, int col, double e) {
		elements[row][col] = e;
	}

	public double get(int row, int col) {
		return elements[row][col];
	}

	public Matrix scalarAddition(double a) {
		Matrix B = new Matrix(this.row, this.col);
		for (int row = 0; row < this.row; row++)
			for (int col = 0; col < this.col; col++)
				B.set(row, col, a + this.get(row, col));
		return B;
	}

	public Matrix scalarSubstraction(double a) {
		Matrix B = new Matrix(this.row, this.col);
		for (int row = 0; row < this.row; row++)
			for (int col = 0; col < this.col; col++)
				B.set(row, col, a - this.get(row, col));
		return B;
	}

	public Matrix scalarMultiplication(double a) {
		Matrix B = new Matrix(this.row, this.col);
		for (int row = 0; row < this.row; row++)
			for (int col = 0; col < this.col; col++)
				B.set(row, col, this.get(row, col) * a);
		return B;
	}

	public Matrix applyFuntion(Function<Double, Double> f) {
		Matrix B = new Matrix(this.row, this.col);
		for (int row = 0; row < this.row; row++)
			for (int col = 0; col < this.col; col++)
				B.set(row, col, f.apply(this.get(row, col)));
		return B;
	}

	public Matrix multByElement(Matrix B) {
		Matrix C = new Matrix(this.row, this.col);
		for (int row = 0; row < this.row; row++)
			for (int col = 0; col < this.col; col++)
				C.set(row, col, this.get(row, col) * B.get(row, col));
		return C;
	}

	public Matrix matrixAddition(Matrix B) {
		if (!(this.row == B.row && this.col == B.col))
			throw new RuntimeException("the matrices dimensions do not add up");

		Matrix C = new Matrix(this.row, this.col);
		for (int row = 0; row < this.row; row++)
			for (int col = 0; col < this.col; col++)
				C.set(row, col, this.get(row, col) + B.get(row, col));
		return C;
	}

	public Matrix matrixSubstraction(Matrix B) {
		if (!(this.row == B.row && this.col == B.col))
			throw new RuntimeException("the matrices dimensions do not add up");

		Matrix C = new Matrix(this.row, this.col);
		for (int row = 0; row < this.row; row++)
			for (int col = 0; col < this.col; col++)
				C.set(row, col, this.get(row, col) - B.get(row, col));
		return C;
	}

	public Matrix matrixMultiplication(Matrix B) {
		if (!(this.col == B.row))
			throw new RuntimeException("the matrices dimensions do not add up: A. col = " + this.col + ", B.row = " + B.row );

		Matrix C = new Matrix(this.row, B.col);
		for (int row = 0; row < this.row; row++)
			for (int col = 0; col < B.col; col++) {
				double sum = 0;
				for (int k = 0; k < this.col; k++)
					sum += this.get(row, k) * B.get(k, col);
				C.set(row, col, sum);
			}
		return C;
	}

	public String toString() {
		StringBuilder str = new StringBuilder();
		for (double row[] : elements) {
			for (double d : row)
				str.append(d + " ");
			str.append("\n");
		}
		return str.toString();
	}
}
