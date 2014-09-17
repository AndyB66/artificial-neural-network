import java.util.ArrayList;
import java.util.ListIterator;



public class NeuralNetwork
{
	
	// #BEGIN: Declarations
	private ArrayList<Layer> layers;
	private ActivationFunction activationFunction;
	
	private double rangeMin;
	private double rangeMax;
	private double range;
	
	private boolean useBias;
	
	// statistics about recent error correction
	private double recentAvgError = Constants.DBL_ZERO;
	private double recentAvgSmoothing = Constants.DBL_ZERO;
	private Double lastError = null;
	
	// #END: Declarations
	
	
	// #BEGIN: Constructors
	
	/**
	 * Initializes an empty neural network with (suggested) input ranges in a given range.
	 * @param activationFunc the activation function to use for neurons in the neural network.
	 * @param rangeMin the lower-bound of the suggested input range.
	 * @param rangeMax the upper-bound of the suggested input range.
	 */
	public NeuralNetwork(final ActivationFunction activationFunc, final double rangeMin, final double rangeMax)
	{
		setLayers(new ArrayList<Layer>());
		useBias = true;
		
		this.activationFunction = activationFunc;
		this.setRange(rangeMin, rangeMax);
	}
	
	/**
	 * Initializes a fully connected neural-network.
	 * @param activationFunc - the activation function to use for neurons in the neural network.
	 * @param topology - the array representing the topology of the neural network where 0 is the input layer and each integer value is the number of neurons in the respective layer.
	 * @param useBias - the boolean determining whether the neural network will employ bias neurons.
	 * @param rangeMin - the lower-bound of the suggested input range.
	 * @param rangeMax - the upper-bound of the suggested input range.
	 * @throws IllegalArgumentException if the topology array length is less than 2.
	 */
	public NeuralNetwork(final ActivationFunction activationFunc, final int[] topology, final boolean useBias, final double rangeMin, final double rangeMax) throws IllegalArgumentException
	{
		this.activationFunction = activationFunc;
		this.useBias = useBias;
		
		this.setRange(rangeMin, rangeMax);
		
		if (topology.length < 2)
			throw new IllegalArgumentException("Cannot initialize NeuralNetwork with less than two layers.");
		
		
		// initialize ArrayList
		setLayers(new ArrayList<Layer>(topology.length));
		
		// fill and connect each layer
		Layer previous = null;
		for (int i = 0; i < topology.length;)
		{
			int current = topology[i];
			
			// create layer (also increments i and checks if last layer (disables bias))
			Layer curLayer = new Layer(current);
			
			// get previous and next layer sizes (i previously incremented for 'outgoing' check)
			int incoming = (previous == null ? 0 : previous.getNeurons().size());
			int outgoing = (i == topology.length ? 0 : topology[i]);
			
			// create neurons
			for (int j = 0; j < current; j++)
			{
				Neuron curNeuron = new Neuron(incoming, outgoing);
				
				// connect previous layer neurons to new Neuron
				if (previous != null)
				{
					for (Neuron n : previous.getNeurons())
						new Connection().connect(n, curNeuron);
					
					
					// connect bias
					if (useBias)
						new Connection().connect(previous.getBiasNeuron(), curNeuron);
				}
				
				curLayer.getNeurons().add(curNeuron);
			}
			
			this.getLayers().add(curLayer);
			previous = curLayer;
		}
		
	}

	/**
	 * Initializes a fully connected neural-network.
	 * @param activationFunc the activation function to use for neurons in the neural network.
	 * @param topology the array representing the topology of the neural network where 0 is the input layer and each integer value is the number of neurons in the respective layer.
	 * @param useBias the boolean determining whether the neural network will employ bias neurons.
	 * @param rangeMin the lower-bound of the suggested input range.
	 * @param rangeMax the upper-bound of the suggested input range.
	 * @param lastError the last error value to initialize the neural network with.
	 * @param recentAvgError the recent average error to initialize the neural network with.
	 * @param recentAvgSmoothing the recent average smoothing factor to initialize the neural network with.
	 */
	public NeuralNetwork(final ActivationFunction activationFunc, final int[] topology, final boolean useBias, final double rangeMin, final double rangeMax, final double lastError, final double recentAvgError, final double recentAvgSmoothing)
	{
		// call previous constructor
		this(activationFunc, topology, useBias, rangeMin, rangeMax);
		
		this.lastError = lastError;
		this.recentAvgError = recentAvgError;
		this.recentAvgSmoothing = recentAvgSmoothing;
	}
	
	// #END: Constructors
	
	
	// #BEGIN: Feed-Forward
	
	/**
	 * Inputs given <code>inputVals</code> into the neural network and iterates through the network to update all neuron values.
	 * @param inputVals - the values to input into the first layer. Size must match size of first layer.
	 * @throws IllegalArgumentException if the size of inputVals does not match the number of neurons in the first layer.
	 */
	public void feedForward(final double[] inputVals) throws IllegalArgumentException
	{
		ListIterator<Layer> layIter = getLayers().listIterator();
		
		{
			ArrayList<Neuron> firstLayNeurons = layIter.next().getNeurons();
			// check for invalid size of input values
			if (inputVals.length != firstLayNeurons.size())
				throw new IllegalArgumentException("Size of input values array does not match the number of first-layer neurons.");
			
			
			// plug input values into the neural network
			for (int i = 0; i < inputVals.length; i++)
				firstLayNeurons.get(i).setValue(inputVals[i]);
		}
		
		// feed forward the network
		Layer curLay = null;
		
		while (layIter.hasNext())
		{
			curLay = layIter.next();
			
			for (Neuron curNeuron : curLay.getNeurons())
				curNeuron.updateValues(activationFunction);
		}
	}
	
	/**
	 * Returns the output layer of the neural network.
	 * @return the output layer of the neural network as an array of doubles.
	 */
	public double[] fetchResults()
	{
		// get last layer neurons
		ArrayList<Neuron> lastLayNeur = getLayers().get(getLayers().size() - 1).getNeurons();
		
		// initialize ArrayList of results with proper size
		double[] results = new double[lastLayNeur.size()];
		
		for (int i = 0; i < lastLayNeur.size(); i++)
			results[i] = lastLayNeur.get(i).getValue();
		
		return results;
	}
	
	// #END: Feed-Forward
	
	
	// #BEGIN: Training
	
	public void iRPROP(final double[] expectedVals) throws IllegalArgumentException
	{
		Layer outLay = getLayers().get(getLayers().size() - 1);
		
		// check for matching expected and output layer sizes
		if (expectedVals.length != outLay.getNeurons().size())
			throw new IllegalArgumentException("Expected values dimensions do not match number of output-layer neurons.");
		
		double error = Constants.DBL_ZERO;
		
		// calculate error and gradients of output layer
		{
			ArrayList<Neuron> outLayNeurons = outLay.getNeurons();
			
			for (int i = 0; i < expectedVals.length; i++)
			{
				Neuron nextNeur = outLayNeurons.get(i);
				double nextExpVal = expectedVals[i];
				
				// calculate error
				double delta = nextExpVal - nextNeur.getValue();
				error += delta * delta;
				
				// calculate delta & gradients
				nextNeur.updateOutputDelta(activationFunction, nextExpVal);
				nextNeur.updateInputGradients();
			}
		}
		
		
		// divide to get average (account for bias)
		error /= outLay.getNeurons().size();
		
		// get the Root Mean Square
		error = Math.sqrt(error);
		
		// recent average error measurement
		recentAvgError = (recentAvgError * recentAvgSmoothing + error) / 
				(recentAvgSmoothing + 1.0);
	
		
		// calculate hidden layer errors and gradients
		{
			// get iterator from second to last layer
			ListIterator<Layer> layIter = getLayers().listIterator(getLayers().size() - 2);
			
			while (layIter.previousIndex() > 0)
			{
				ArrayList<Neuron> neurons = layIter.previous().getNeurons();
				
				for (Neuron n : neurons)
				{
					n.updateHiddenDelta(activationFunction);
					n.updateInputGradients();
				}
			}
		}
		
		// update all weights
		{
			// get iterator from second to last layer
			ListIterator<Layer> layIter = getLayers().listIterator(getLayers().size() - 2);
			
			while (layIter.previousIndex() > 0)
			{
				ArrayList<Neuron> neurons = layIter.previous().getNeurons();
				
				for (Neuron n : neurons)
					n.updateInputWeights(error, this.lastError, Constants.MIN_DELTA, Constants.MAX_DELTA);
				
			}
		}
		
		// set last error to this error
		this.lastError = error;
	}
	
	public void train(final double[][] data, final double[][] expectedVals)
	{
		int min = Math.min(data.length, expectedVals.length);
		
		for (int i = 0; i < min; i++)
		{
			feedForward(data[i]);
			iRPROP(expectedVals[i]);
		}
	}
	
	// #END: Training
	
	
	// #BEGIN: Normalize
	
	public int normalize(final int val, final int origMin)
	{
		return (int)Math.round((val - origMin) / (range));
	}
	
	public int[] normalize(final int vals[], final int origMin)
	{
		for (int i = 0; i < vals.length; i++)
			vals[i] = normalize(vals[i], origMin);
		
		return vals;
	}
	
	public int[] normalize(final int vals[], final int origMin[])
	{
		for (int i = 0; i < vals.length; i++)
		{
			vals[i] = normalize(vals[i], origMin[i]);
		}
		
		return vals;
	}
	
	public double normalize(final double val, final double origMin)
	{
		return (val - origMin) / range;
	}
	
	public double[] normalize(final double vals[], final double origMin)
	{
		for (int i = 0; i < vals.length; i++)
			vals[i] = normalize(vals[i], origMin);
		
		return vals;
	}
	
	public double[] normalize(final double vals[], final double origMin[])
	{
		for (int i = 0; i < vals.length; i++)
		{
			vals[i] = normalize(vals[i], origMin[i]);
		}
		
		return vals;
	}
	
	// #END: Normalize
	
	
	// #BEGIN: Fields
	
	public ArrayList<Layer> getLayers()
	{
		return layers;
	}

	private void setLayers(final ArrayList<Layer> layers)
	{
		this.layers = layers;
	}

	public double getRangeMin()
	{
		return rangeMin;
	}

	public double getRangeMax()
	{
		return rangeMax;
	}
	
	public void setRange(final double rangeMin, final double rangeMax)
	{
		// if inputted min > max, swap min/max
		if (rangeMin > rangeMax)
		{
			this.rangeMin = rangeMax;
			this.rangeMax = rangeMin;
		}
		else
		{
			this.rangeMin = rangeMin;
			this.rangeMax = rangeMax;
		}
		
		this.range = this.rangeMax - this.rangeMin;
	}

	public boolean isUseBias()
	{
		return useBias;
	}

	public void setUseBias(final boolean useBias)
	{
		this.useBias = useBias;
	}

	public ActivationFunction getActivationFunction()
	{
		return activationFunction;
	}
	
	public Double getLastError()
	{
		return this.lastError;
	}
	
	// #END: Fields
	
}