package reneural;

import java.io.FileNotFoundException;

public class Main {
	private static double treino[][] = {
			{0, 0, 1, 1},
			{0, 1, 0, 1},
			{-1, -1, -1, -1}
		};
		
		private static double esperado[] = {0, 1, 1, 0};
		
		public static void main(String[] args) throws FileNotFoundException {
			double[][] x = new double[10][];
			x[0] = new double[5];
			x[1] = new double[7];
			x[2] = new double[2];
			
			
			
			int qtdeNeuroniosCamadaIntermediaria = 2;
			int qtdeNeuroniosEntrada = 3;

			RedeNeural rede = new RedeNeural(qtdeNeuroniosCamadaIntermediaria, qtdeNeuroniosEntrada);
			
			System.out.println("Treinando...");
			rede.treinar(treino, esperado);
			
			System.out.println("Teste:");
			rede.classificar(new double[] { 0, 0,-1 });
		}
}
