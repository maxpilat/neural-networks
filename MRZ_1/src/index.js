/*
Лабораторная работа №1 по дисциплине МРЗВИС
Выполнена студентом группы 121702 БГУИР Пилатом Максимом Дмитриевичем
Вариант 13: Реализовать модель линейной рециркуляционной сети с постоянным коэффициентом обучения с ненормированными весами
*/

const RECT_SIZE = 8;
const INPUT_SIZE = RECT_SIZE * RECT_SIZE * 3;
const OUTPUT_SIZE = 128;
const IMAGE_SIZE = 256;
const BITS_PER_BYTE = 8;

document.getElementById("imageInput").addEventListener("change", function (event) {
  const file = event.target.files[0];

  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      const img = new Image();
      img.src = e.target.result;

      img.onload = function () {
        const canvas = document.getElementById("canvas");
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        new App(img, canvas, RECT_SIZE, IMAGE_SIZE, INPUT_SIZE, OUTPUT_SIZE).run();
      };
    };

    reader.readAsDataURL(file);
  }
});

class App {
  constructor(image, canvas, rectSize, imageSize, inputSize, outputSize) {
    this.image = image;
    this.canvas = canvas;
    this.rectSize = rectSize;
    this.imageSize = imageSize;
    this.inputSize = inputSize;
    this.outputSize = outputSize;
  }

  run() {
    const blocks = ImageSolver.split(this.image, this.canvas, this.rectSize);
    this.neuralNetwork = new NeuralNetwork(0.0001, 3500, this.inputSize, this.outputSize);
    this.neuralNetwork.train(blocks);

    let rects = [];

    for (let bitMap of blocks) {
      let rect = this.neuralNetwork.getRestoredValue(bitMap);
      rects.push(rect[0]);
    }

    const compressedImage = ImageSolver.restore(
      rects,
      this.rectSize,
      this.imageSize,
      this.imageSize,
      document.getElementById("canvas2")
    );
    const compressionInfoSize =
      (compressedImage.length * BITS_PER_BYTE +
        this.inputSize * BITS_PER_BYTE +
        this.canvas.width * this.canvas.height * BITS_PER_BYTE +
        this.rectSize * this.rectSize * BITS_PER_BYTE) *
      BITS_PER_BYTE;
    const imageSize = this.imageSplitter.blocks.reduce((acc, el) => acc + Math.pow(el.length, 2), 0);

    document.getElementById("koeff").append((imageSize * BITS_PER_BYTE) / compressionInfoSize);
  }
}

class NeuralNetwork {
  firstLayerWeights = [];
  secondLayerWeights = [];

  constructor(learningRate, errorThreshold, inputSize, outputSize) {
    this.learningRate = learningRate;
    this.errorThreshold = errorThreshold;

    this.firstLayerWeights = this.generateRandomWeights(inputSize, outputSize);
    this.secondLayerWeights = MatrixSolver.transpose(this.firstLayerWeights);
  }

  train(vectorsOfColor) {
    let error = 999999;
    let errorsPerIter = 0;
    while (error > this.errorThreshold) {
      for (let vectorOfColor of vectorsOfColor) {
        let feedForwardedMatrices = this.feedForward(vectorOfColor);
        errorsPerIter += this.backPropagate(vectorOfColor, feedForwardedMatrices);
      }

      error = errorsPerIter;
      console.log(error);
      errorsPerIter = 0;
    }
  }

  getRestoredValue(vectorOfColor) {
    return this.feedForward(vectorOfColor).secondLayerMatrix;
  }

  feedForward(vectorOfColor) {
    vectorOfColor = [vectorOfColor];
    let firstLayerMatrix = MatrixSolver.multiply(vectorOfColor, this.firstLayerWeights);
    let secondLayerMatrix = MatrixSolver.multiply(firstLayerMatrix, this.secondLayerWeights);

    return {
      firstLayerMatrix,
      secondLayerMatrix,
    };
  }

  backPropagate(vectorOfColor, forwardedMatrices) {
    let { firstLayerMatrix, secondLayerMatrix } = forwardedMatrices;

    vectorOfColor = [vectorOfColor];

    let deltaLastLayer = MatrixSolver.subtract(secondLayerMatrix, vectorOfColor);

    // w_1новое = w_1текущее-error_1 * входное_значение_1 * кэф_обуч
    this.secondLayerWeights = MatrixSolver.subtract(
      this.secondLayerWeights,
      MatrixSolver.transpose(
        MatrixSolver.multiplyByNumber(
          MatrixSolver.multiply(MatrixSolver.transpose(deltaLastLayer), firstLayerMatrix),
          this.learningRate
        )
      )
    );

    let deltaFirstLayer = MatrixSolver.multiply(deltaLastLayer, MatrixSolver.transpose(this.secondLayerWeights));

    this.firstLayerWeights = MatrixSolver.subtract(
      this.firstLayerWeights,
      MatrixSolver.multiplyByNumber(
        MatrixSolver.multiply(MatrixSolver.transpose(vectorOfColor), deltaFirstLayer),
        this.learningRate
      )
    );

    return deltaLastLayer.reduce((acc, row) => acc + row.reduce((sum, val) => sum + val ** 2, 0), 0);
  }

  generateRandomWeights(inputSize, outputSize, minValue = -1, maxValue = 1) {
    const range = maxValue - minValue;

    return Array.from({ length: inputSize }, () =>
      Array.from({ length: outputSize }, () => Math.random() * range + minValue)
    );
  }

  normalizeVector(vector) {
    let sum = 0;

    for (let i = 0; i < vector.length; i++) {
      let c = vector[i];
      sum += c * c;
    }

    let sqrSum = Math.sqrt(sum);

    if (sqrSum === 0) {
      return vector;
    }

    return vector.map((value) => value / sqrSum);
  }

  normalize(vector) {
    let sum = 0;

    for (let i = 0; i < vector.length; i++) {
      for (let j = 0; j < vector[i].length; j++) {
        let c = vector[i][j];
        sum += c * c;
      }
    }

    let sqrSum = Math.sqrt(sum);

    if (sqrSum === 0) {
      return vector;
    }

    return vector.map((row) => row.map((value) => value / sqrSum));
  }
}

class ImageSolver {
  static split(image, canvas, rectSize) {
    const ctx = canvas.getContext("2d");
    const blocks = [];

    for (let y = 0; y < image.height; y += rectSize) {
      for (let x = 0; x < image.width; x += rectSize) {
        const imageData = ctx.getImageData(x, y, rectSize, rectSize);
        const filteredRow = [];
        for (let i = 0; i < imageData.data.length; i += 4) {
          filteredRow.push((imageData.data[i] * 2) / 255 - 1); // Red
          filteredRow.push((imageData.data[i + 1] * 2) / 255 - 1); // Green
          filteredRow.push((imageData.data[i + 2] * 2) / 255 - 1); // Blue
        }
        blocks.push(filteredRow);
      }
    }
    return blocks;
  }

  static restore(bitMaps, blockSize, width, height, canvas) {
    const ctx = canvas.getContext("2d");
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    let blockIndex = 0;
    for (let y = 0; y < height; y += blockSize) {
      for (let x = 0; x < width; x += blockSize) {
        const block = bitMaps[blockIndex];

        for (let i = 0; i < blockSize; i++) {
          for (let j = 0; j < blockSize; j++) {
            if (y + i < height && x + j < width) {
              const pixelIndex = (i * blockSize + j) * 3; // Индекс R в block
              const dataIndex = ((y + i) * width + (x + j)) * 4; // Индекс R в data

              data[dataIndex] = Math.round(((block[pixelIndex] + 1) * 255) / 2); // R
              data[dataIndex + 1] = Math.round(((block[pixelIndex + 1] + 1) * 255) / 2); // G
              data[dataIndex + 2] = Math.round(((block[pixelIndex + 2] + 1) * 255) / 2); // B
              data[dataIndex + 3] = 255; // Alpha
            }
          }
        }
        blockIndex++;
      }
    }

    ctx.putImageData(imageData, 0, 0);

    return data;
  }
}

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

  static subtract(matrixA, matrixB) {
    const rowsA = matrixA.length;
    const colsA = matrixA[0].length;
    const rowsB = matrixB.length;
    const colsB = matrixB[0].length;

    if (rowsA !== rowsB || colsA !== colsB) {
      throw new Error("Размеры матриц должны совпадать для выполнения вычитания.");
    }

    const result = new Array(rowsA).fill(null).map(() => new Array(colsA).fill(0));

    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsA; j++) {
        result[i][j] = matrixA[i][j] - matrixB[i][j];
      }
    }

    return result;
  }

  static multiplyByNumber(matrix, num) {
    return matrix.map((row) => row.map((el) => el * num));
  }
}
