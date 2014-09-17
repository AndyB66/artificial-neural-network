

class Connection
{
	
	// #BEGIN: Declarations
	private Neuron fromNeuron;
	private Neuron toNeuron;
	
	private double weight;
	private double lastWeightChange = Constants.DBL_ZERO;
	
	private double gradient = Constants.DBL_ZERO;
	private double lastGradient = Constants.DBL_ZERO;
	
	private double learnDelta = Constants.INIT_LEARN_DELTA;
	// #END: Declarations
	
	
	// #BEGIN: Public Functions
	
	public static double randomWeight()
	{
		return new java.util.Random().nextDouble();
	}
	
	public Connection()
	{
		this.weight = Connection.randomWeight();
	}
	
	public Connection(final double weight, final double lastWeightChange, final double gradient, final double lastGradient, final double learnDelta)
	{
		this.weight = weight;
		this.lastWeightChange = lastWeightChange;
		
		this.gradient = gradient;
		this.lastGradient = lastGradient;
		
		this.learnDelta = learnDelta;
	}
	
	public void connect(final Neuron from, final Neuron to)
	{
		from.addOut(this);
		to.addIn(this);
	}

	// #END: Public Functions

	
	// #BEGIN: Fields
	
	public Neuron getFromNeuron()
	{
		return fromNeuron;
	}

	public void setFromNeuron(final Neuron fromNeuron)
	{
		this.fromNeuron = fromNeuron;
	}

	public Neuron getToNeuron()
	{
		return toNeuron;
	}

	public void setToNeuron(final Neuron toNeuron)
	{
		this.toNeuron = toNeuron;
	}

	public double getWeight()
	{
		return weight;
	}

	public void setWeight(final double weight)
	{
		this.weight = weight;
	}

	public double getLastWeightChange()
	{
		return lastWeightChange;
	}

	public void setLastWeightChange(final double lastWeightChange)
	{
		this.lastWeightChange = lastWeightChange;
	}

	public double getGradient()
	{
		return gradient;
	}

	public void setGradient(final double gradient)
	{
		this.gradient = gradient;
	}

	public double getLastGradient()
	{
		return lastGradient;
	}

	public void setLastGradient(final double lastGradient)
	{
		this.lastGradient = lastGradient;
	}

	public double getLearnDelta()
	{
		return learnDelta;
	}

	public void setLearnDelta(final double learnDelta)
	{
		this.learnDelta = learnDelta;
	}
	
	// #END: Fields
}
