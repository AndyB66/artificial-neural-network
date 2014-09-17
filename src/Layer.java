import java.util.ArrayList;
import java.util.Iterator;

class Layer
{
	
	// #BEGIN: Declarations
	private ArrayList<Neuron> neurons;
	private final Neuron biasNeuron = new Neuron(0, 8, 1.0);
	// #END: Declarations
	
	
	// #BEGIN: Public Functions
	
	public Layer()
	{
		neurons = new ArrayList<Neuron>();
	}
	
	public Layer(final int size)
	{
		neurons = new ArrayList<Neuron>(size);
	}
	
	// #END: Public Functions
	
	
	// #BEGIN: Fields
	
	public Neuron getBiasNeuron()
	{
		return biasNeuron;
	}

	public ArrayList<Neuron> getNeurons()
	{
		return neurons;
	}

	public Iterator<Neuron> iterator()
	{
		return neurons.iterator();
	}
	
	// #END: Fields
}
