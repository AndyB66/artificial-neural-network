import java.util.ArrayList;

class Neuron
{
	
	// #BEGIN: Declarations
	private ArrayList<Connection> connIn;
	private ArrayList<Connection> connOut;
	
	private double summation = Constants.DBL_ZERO;
	private double value = Constants.DBL_ZERO;
	
	
	private double delta = Constants.DBL_ZERO;
	// #END: Declarations
	
	
	// #BEGIN: Constructors
	
	public Neuron()
	{
		connIn = new ArrayList<Connection>(8);
		connOut = new ArrayList<Connection>(8);
		
		this.value = Constants.DBL_ZERO;
		
		this.delta = Constants.DBL_ZERO;
	}
	
	public Neuron(final int incomingCount, final int outgoingCount)
	{
		connIn = new ArrayList<Connection>(incomingCount);
		connOut = new ArrayList<Connection>(outgoingCount);
	}
	
	public Neuron(final int incomingCount, final int outgoingCount, final double initValue)
	{
		connIn = new ArrayList<Connection>(incomingCount);
		connOut = new ArrayList<Connection>(outgoingCount);
		
		this.value = initValue;
	}
	
	// #END: Constructors
	
	
	// #BEGIN: Add/Remove
	
	public void addIn(Connection addConn)
	{
		addConn.setToNeuron(this);
		connIn.add(addConn);
	}
	
	public boolean removeIn(Connection remConn)
	{
		remConn.setToNeuron(null);
		return connIn.remove(remConn);
	}
	
	public void addOut(Connection addConn)
	{
		addConn.setFromNeuron(this);
		connOut.add(addConn);
	}
	
	public boolean removeOut(Connection remCon)
	{
		remCon.setFromNeuron(null);
		return connOut.remove(remCon);
	}
	
	// #END: Add/Remove
	
	
	// #BEGIN: Update Functions
	
	public void updateValues(final ActivationFunction func)
	{
		this.summation = calcSummation();
		this.value = ActivationFunction.calcActivation(func, this.summation);
	}
	
	public void updateOutputDelta(final ActivationFunction func, final double targetVal)
	{
		double diff = targetVal - this.getValue();
		this.delta = diff * ActivationFunction.calcDerivative(func, this.summation);
	}
	
	public void updateHiddenDelta(final ActivationFunction func)
	{
		double sum = Constants.DBL_ZERO;
		
		for (Connection c : connOut)
			sum += (c.getWeight() * c.getToNeuron().delta);
		
		this.delta = sum * ActivationFunction.calcDerivative(func, this.summation);
	}
	
	public void updateInputGradients()
	{
		for (Connection c : connIn)
			c.setGradient(this.delta * c.getFromNeuron().getValue());
	}
	
	public void updateInputWeights(double error, Double lastError, double minDelta, double maxDelta)
	{
		for (Connection c : connIn)
		{
			final int change = (int)Math.signum(c.getGradient() * c.getLastGradient());
			double weightChange = Constants.DBL_ZERO;
			
			switch (change)
			{
				// Still heading in the correct direction
				case 1:
				{
					double gradient = c.getGradient();
					
					// calculate learn delta
					double learnDelta = Math.min(c.getLearnDelta() * Constants.POS_ETA, maxDelta);
					
					// calculate new delta for next iteration
					c.setLearnDelta(learnDelta);
					
					// continue in same direction with weight change
					weightChange = -Math.signum(gradient) * learnDelta;
					
					// set current gradient as last gradient
					c.setLastGradient(gradient);
					break;
				}
				
				// Last step crossed over minimum, reverse direction
				case -1:
				{
					// calculate delta
					c.setLearnDelta(Math.max(c.getLearnDelta() * Constants.NEG_ETA, minDelta));
					
					// if created bigger mess, undo everything
					if (lastError != null && error > lastError)
						weightChange = -c.getLastWeightChange();
					
					// set last gradient to 0 to avoid double jeopardy
					c.setLastGradient(Constants.DBL_ZERO);
					break;
				}
				
				// very close to accurate reading
				case 0:
				{
					double gradient = c.getGradient();
					
					// make no changes to learn delta
					weightChange = -Math.signum(gradient) * c.getLearnDelta();
					
					// set current gradient as last gradient
					c.setLastGradient(gradient);
					break;
				}
			}
			
			// set new weight
			c.setWeight(c.getWeight() + weightChange);
			
			// set last weight change
			c.setLastWeightChange(weightChange);
		}
	}
	
	// #END: Update Functions
	
	
	// #BEGIN: Private Functions
	
	private double calcSummation()
	{
		double sum = Constants.DBL_ZERO;
		for (Connection conn : connIn)
			sum += conn.getFromNeuron().getValue() * conn.getWeight();
		
		return sum;
	}
	
	// #END: Private Functions

	
	// #BEGIN: Fields

	public double getValue()
	{
		return value;
	}

	public void setValue(final double value)
	{
		this.value = value;
	}
	
	// #END: Fields
}
