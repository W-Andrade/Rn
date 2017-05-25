package reneural;

import java.util.Arrays;

public class RedeNeural {
	private static double TAXA_APRENDIZADO = 0.03;

	private double[][] primeiraCamada;
	private double[] segundaCamada;
	private int qtdePrimeiraCamada;
	private int qtdeNeuroniosEntrada;
	private int epocas = 0;

	public RedeNeural(int qtdePrimeiraCamada, int qtdeEntrada) {
		this.qtdePrimeiraCamada = qtdePrimeiraCamada;
		this.qtdeNeuroniosEntrada = qtdeEntrada;
		this.inicializarPesos();
	}

	public void treinar(double[][] treinamento, double[] esperados) {
		double erro = 1.0;
		while ((Math.abs(erro) > 0.05) && (epocas < 100000)) {
			for (int i = 0; i < treinamento[0].length; i++) {
				double[] entradaSegundaCamada = propagarSinalPelaPrimeiraCamada(treinamento, i);
				double valorSaida = propagarSinalPelaSegundaCamada(entradaSegundaCamada);
				erro = calcularErro(esperados, valorSaida, i);
				double gradiente = gradienteDeRetropopagacao(valorSaida, erro);
				aprender(treinamento, entradaSegundaCamada, gradiente, i);
			}
			epocas++;
		}
	}

	public void classificar(double[] entrada) {
		if (epocas > 99999) {
			System.out.println("Nao foi possivel atingir um ponto de convergencia, verifique os parametros e a estrutura da rede.");
		} else {
			double[] saidasPrimeiraCamada = saidaClassificacaoPrimeiraCamada(entrada);
			double[] entradaSegundaCamada = entradasSegundaCamada(saidasPrimeiraCamada);
			double y = propagarSinalPelaSegundaCamada(entradaSegundaCamada);
			long value = Math.round(y);
			System.out.println(value);
		}
	}

	private void aprender(double[][] conjuntoTreinamento, double[] entradaSegundaCamada, double gradiente, int i) {
		retropropagarErroPelaSegundaCamada(entradaSegundaCamada, gradiente);
		retropropagarErroPelaPrimeiraCamada(conjuntoTreinamento, entradaSegundaCamada, gradiente, i);
	}

	private double[] propagarSinalPelaPrimeiraCamada(double[][] conjuntoTreinamento, int i) {
		double[] saidasPrimeiraCamada = saidaTreinamentoPrimeiraCamada(conjuntoTreinamento, i);
		return entradasSegundaCamada(saidasPrimeiraCamada);
	}

	private double propagarSinalPelaSegundaCamada(double[] entradaSegundaCamada) {
		double u = 0;
		for (int j = 0; j < segundaCamada.length; j++) {
			u += entradaSegundaCamada[j] * segundaCamada[j];
		}
		return funcaoTransferencia(u);
	}

	private double[] entradasSegundaCamada(double[] saidasPrimeiraCamada) {
		double[] entradaSegundaCamada = Arrays.copyOf(saidasPrimeiraCamada, saidasPrimeiraCamada.length + 1);
		entradaSegundaCamada[entradaSegundaCamada.length - 1] = 1.0;
		return entradaSegundaCamada;
	}

	private double[] saidaTreinamentoPrimeiraCamada(double[][] conjuntoTreinamento, int i) {
		double[] saidasPrimeiraCamada = new double[qtdePrimeiraCamada];
		for (int j = 0; j < primeiraCamada.length; j++) {
			double u = 0;
			for (int k = 0; k < primeiraCamada[j].length; k++) {
				u += conjuntoTreinamento[k][i] * primeiraCamada[j][k];
			}
			saidasPrimeiraCamada[j] = funcaoTransferencia(u);
		}
		return saidasPrimeiraCamada;
	}

	private double[] saidaClassificacaoPrimeiraCamada(double[] entrada) {
		double[] saidasPrimeiraCamada = new double[qtdePrimeiraCamada];
		for (int j = 0; j < primeiraCamada.length; j++) {
			double u = 0;
			for (int k = 0; k < primeiraCamada[j].length; k++) {
				u += entrada[k] * primeiraCamada[j][k];
			}
			saidasPrimeiraCamada[j] = funcaoTransferencia(u);
		}
		return saidasPrimeiraCamada;
	}

	private void retropropagarErroPelaPrimeiraCamada(double[][] conjuntoTreinamento, double[] entradaSegundaCamada, double gradiente, int i) {
		for (int j = 0; j < entradaSegundaCamada.length - 1; j++) {
			double derivadaFuncaoTransferencia = entradaSegundaCamada[j] * (1.0 - entradaSegundaCamada[j]);
			double sigma = derivadaFuncaoTransferencia * (segundaCamada[j] * gradiente);
			for (int k = 0; k < primeiraCamada[j].length; k++) {
				primeiraCamada[j][k] += RedeNeural.TAXA_APRENDIZADO * sigma * conjuntoTreinamento[k][i];
			}
		}
	}

	private void retropropagarErroPelaSegundaCamada(double[] entradaSegundaCamada, double gradiente) {
		for (int j = 0; j < segundaCamada.length; j++) {
			segundaCamada[j] += RedeNeural.TAXA_APRENDIZADO * entradaSegundaCamada[j] * gradiente;
		}
	}

	private double gradienteDeRetropopagacao(double valorSaida, double erro) {
		return valorSaida * (1 - valorSaida) * erro;
	}

	private double funcaoTransferencia(double u) {
		return 1.0 / (1.0 + Math.exp(-u));
	}

	private double calcularErro(double[] valoresEsperados, double valorSaida, int i) {
		return valoresEsperados[i] - valorSaida;
	}

	private void inicializarPesos() {
		pesosPrimeiraCamada();
		pesosSegundaCamada();
	}

	private void pesosPrimeiraCamada() {
		primeiraCamada = new double[qtdePrimeiraCamada][qtdeNeuroniosEntrada];
		for (int i = 0; i < primeiraCamada.length; i++) {
			for (int j = 0; j < primeiraCamada[i].length; j++) {
				primeiraCamada[i][j] = Math.random();
			}
		}
	}

	private void pesosSegundaCamada() {
		segundaCamada = new double[qtdePrimeiraCamada + 1];
		for (int i = 0; i < segundaCamada.length; i++) {
			segundaCamada[i] = Math.random();
		}
	}

	public void imprimirValoresConexoes() {
		System.out.println("\n Conexoes da primeira camada:");
		for (int i = 0; i < primeiraCamada.length; i++) {
			for (int j = 0; j < primeiraCamada[i].length; j++) {
				System.out.println(primeiraCamada[i][j] + " ");
			}
			System.out.println("\n");
		}

		System.out.println("\n Conexoes da segunda camada:");
		for (int i = 0; i < segundaCamada.length; i++) {
			System.out.println(segundaCamada[i] + " ");
		}

		System.out.println("\n\n");
	}

	public double[][] getConexoesPrimeiraCamada() {
		return primeiraCamada;
	}

	public void setConexoesPrimeiraCamada(double[][] conexoesPrimeiraCamada) {
		this.primeiraCamada = conexoesPrimeiraCamada;
	}

	public double[] getConexoesSegundaCamada() {
		return segundaCamada;
	}

	public void setConexoesSegundaCamada(double[] conexoesSegundaCamada) {
		this.segundaCamada = conexoesSegundaCamada;
	}
}
