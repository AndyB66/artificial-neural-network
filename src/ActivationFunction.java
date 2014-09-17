

public enum ActivationFunction
{
	SIGMOID, TANH, QUICKTANH;
	
	public static double calcActivation(final ActivationFunction func, final double x)
	{
		switch (func)
		{
			case SIGMOID:
				return ActivationFunction.sigmoid(x);
			case TANH:
				return ActivationFunction.tanh(x);
			case QUICKTANH:
				return ActivationFunction.tanh(x);
		}
		
		// fallback case
		return sigmoid(x);
	}
	
	public static double calcDerivative(final ActivationFunction func, final double x)
	{
		switch (func)
		{
			case SIGMOID:
				return ActivationFunction.sigmoidDerivative(x);
			case TANH:
				return ActivationFunction.tanhDerivative(x);
			case QUICKTANH:
				return ActivationFunction.tanhDerivativeQuick(x);
		}
		
		// fallback case
		return sigmoidDerivative(x);
	}
	
	
	// #BEGIN: Natural Logarithmic Sigmoid
	
	/**
	 * Evaluates the sigmoid function at a given x-value. The function will return constants for values not in range [-10, 10].
	 * @param x the value to evaluate the sigmoid function at.
	 * @return the result of the sigmoid function at the given value.
	 */
	private static double sigmoid(final double x)
	{
		return 1 /  (1 + Math.exp(-x));
	}
	
	/**
	 * Evaluates the derivative of the sigmoid function at a given x-value.
	 * @param x the value to evaluate the derivative at.
	 * @return the result of the derivative of the sigmoid function at the given value.
	 */
	private static double sigmoidDerivative(final double x)
	{
		return sigmoidDerivativeFromSigmoid(sigmoid(x));
	}
	
	/**
	 * Evaluates the derivative of the sigmoid function at a given y-value of the sigmoid function.
	 * @param y the y-value of the sigmoid function to find derivative of.
	 * @return the derivative of the sigmoid function at the given y-value.
	 */
	private static double sigmoidDerivativeFromSigmoid(final double y)
	{
		return y * (1 - y);
	}
	
	// #END: Natural Logarithmic Sigmoid

	
	// #BEGIN: Hyperbolic Tangent
	
	private static double tanh(final double x)
	{
		return Math.tanh(x);
	}
	
	private static double tanhDerivative(final double x)
	{
		double calcTanh = Math.tanh(x);
		return 1 - (calcTanh * calcTanh);
	}
	
	private static double tanhDerivativeQuick(final double x)
	{
		return 1 - (x * x);
	}
	
	// #END: Hyperbolic Tangent
}
