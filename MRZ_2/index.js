/*
Лабораторная работа №2 по дисциплине МРЗВИС
Выполнена студентом группы 121702 БГУИР Пилатом Максимом Дмитриевичем
Вариант 13: Реализовать сеть Хопфилда работающую в асинхронном режиме с непрерывным состоянием
Источник https://habr.com/ru/articles/561198/
https://github.com/z0rats/js-neural-networks/tree/master/hopfield_network
*/

class MatrixSolver {
  static transpose(matrix) {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const transposed = [];

    for (let i = 0; i < cols; i++) {
      transposed[i] = [];
      for (let j = 0; j < rows; j++) {
        transposed[i][j] = matrix[j][i];
      }
    }

    return transposed;
  }

  static multiply(matrixA, matrixB) {
    const rowsA = matrixA.length;
    const colsA = matrixA[0].length;
    const rowsB = matrixB.length;
    const colsB = matrixB[0].length;

    if (colsA !== rowsB) {
      throw new Error("Количество столбцов первой матрицы должно быть равно количеству строк второй матрицы.");
    }

    const result = new Array(rowsA).fill(null).map(() => new Array(colsB).fill(0));

    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsB; j++) {
        for (let k = 0; k < colsA; k++) {
          result[i][j] += matrixA[i][k] * matrixB[k][j];
        }
      }
    }

    return result;
  }

  static multiplyByNumber(matrix, num) {
    return matrix.map((row) => row.map((el) => el * num));
  }

  static add(matrixA, matrixB) {
    if (matrixA.length !== matrixB.length || matrixA[0].length !== matrixB[0].length) {
      throw new Error("Матрицы должны быть одинакового размера");
    }

    return matrixA.map((row, i) => row.map((value, j) => value + matrixB[i][j]));
  }

  static subtract(matrixA, matrixB) {
    if (matrixA.length !== matrixB.length || matrixA[0].length !== matrixB[0].length) {
      throw new Error("Размеры матриц должны совпадать для выполнения вычитания.");
    }

    return matrixA.map((row, i) => row.map((value, j) => value - matrixB[i][j]));
  }
}

function imageBeautifulPrint(image, rows, cols) {
  for (let i = 0; i < rows; i++) {
    const row = image
      .slice(i * cols, (i + 1) * cols)
      .map((value) => (Math.sign(value) === 1 ? " # " : " O "))
      .join("");
    console.log(row);
  }
}

class HopfieldNetwork {
  constructor(images, learningRate = 1) {
    this.size = images[0].length;
    this.weights = Array.from({ length: this.size }, () => Array(this.size).fill(0));
    this.images = images;
    this.negImages = this.getNegImages(this.images);
    this.learningRate = learningRate;
    this.iters = 0;
  }

  getNegImages(images) {
    return images.map((image) => image.map((value) => -value));
  }

  train(e = 1e-6, maxIters = 10000) {
    for (let i = 0; i < maxIters; i++) {
      this.iters = i;
      const oldWeights = this.weights.map((row) => [...row]);

      for (let image of this.images) {
        const transposedImage = MatrixSolver.transpose([image]);
        const activation = MatrixSolver.multiply(this.weights, transposedImage).map((arr) =>
          arr.map((value) => Math.tanh(value))
        );
        this.weights = MatrixSolver.add(
          this.weights,
          MatrixSolver.multiplyByNumber(
            MatrixSolver.multiply(
              MatrixSolver.subtract(transposedImage, activation),
              MatrixSolver.transpose(transposedImage)
            ),
            this.learningRate / this.size
          )
        );

        for (let i = 0; i < this.weights.length; i++) {
          this.weights[i][i] = 0;
        }
      }

      let diffSum = 0;

      for (let i = 0; i < oldWeights.length; i++) {
        for (let j = 0; j < oldWeights[i].length; j++) {
          diffSum += Math.abs(oldWeights[i][j] - this.weights[i][j]);
        }
      }

      if (diffSum < e) {
        break;
      }
    }

    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i][i] = 0;
    }
  }

  findImageNum(x, images) {
    for (let imgIndex = 0; imgIndex < images.length; imgIndex++) {
      const image = images[imgIndex];
      const maxDiff = Math.max(...image.map((val, i) => Math.abs(val - x[i])));

      if (maxDiff < 1e-2) {
        return imgIndex; // Возвращает индекс изображения, если найдено совпадение
      }
    }
    return null; // Если не найдено совпадение
  }

  predict(x, maxIters = 1000) {
    let states = Array(4).fill(x.slice());
    let relaxationIters = 0;

    for (let i = 0; i < maxIters; i++) {
      relaxationIters += 1;
      console.log(states, "STATES");
      let newState = MatrixSolver.transpose(
        MatrixSolver.multiply(this.weights, MatrixSolver.transpose([states[states.length - 1]]), true)
      ).map((arr) => arr.map((value) => Math.tanh(value)));
      states.push(...newState);
      states.shift();

      if (i >= 3 && this.findAbsMax(0, 2, states) < 1e-8 && this.findAbsMax(1, 3, states) < 1e-8) {
        let imageNum = this.findImageNum(newState, this.images);
        let negImageNum = this.findImageNum(newState, this.negImages);
        let isNegative = negImageNum !== null;

        return {
          relaxationIters,
          newState,
          imageNum: imageNum !== null ? imageNum : negImageNum,
          isNegative,
        };
      }
    }

    return {
      relaxationIters: maxIters,
      newState,
      imageNum: null,
      isNegative: null,
    };
  }

  findAbsMax(first, second, states) {
    let difference = states[first].map((val, index) => Math.abs(val - states[second][index]));

    return Math.max(...difference);
  }
}

let alphabet = [
  [
    [-1, 1, 1, -1],
    [1, -1, -1, 1],
    [1, 1, 1, 1],
    [1, -1, -1, 1],
  ],
  [
    [1, 1, 1, -1],
    [1, -1, -1, 1],
    [1, -1, -1, 1],
    [1, 1, 1, -1],
  ],
  [
    [-1, 1, 1, -1],
    [1, -1, -1, 1],
    [1, -1, -1, 1],
    [-1, 1, 1, -1],
  ],
  [
    [-1, 1, 1, 1],
    [1, -1, -1, -1],
    [1, -1, -1, -1],
    [-1, 1, 1, 1],
  ],
];

alphabet = alphabet.map((item) => (Array.isArray(item) && Array.isArray(item[0]) ? item.flat() : item));

const network = new HopfieldNetwork(alphabet, 0.7);
network.train();

const testImage = [-1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1];

const { relaxationIters, newState: state, imageNum: imageIdx, isNegative } = network.predict(testImage, 10000);

let predictedImage;

if (imageIdx) {
  predictedImage = isNegative ? network.negImages[imageIdx] : network.images[imageIdx];
} else {
  predictedImage = state;
}

console.log("All images");
for (let img of alphabet) {
  imageBeautifulPrint(img.flat(), 4, 4);
  console.log();
}

console.log("INPUT IMAGE");
imageBeautifulPrint(testImage, 4, 4);

console.log("PREDICTED IMAGE");
imageBeautifulPrint(...predictedImage, 4, 4);
