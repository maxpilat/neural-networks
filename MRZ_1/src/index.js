/*
Лабораторная работа №1 по дисциплине МРЗВИС
Выполнена студентом группы 121702 БГУИР Пилатом Максимом Дмитриевичем, программный код заимствован у Летко Александра Юрьевича
Вариант 13: Реализовать модель линейной рециркуляционной сети с постоянным коэффициентом обучения с нормированными весами
*/

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

        new App(img, canvas).run();
      };
    };

    reader.readAsDataURL(file);
  }
});

const INPUT_SIZE = 192;
const RECT_SIZE = 8;
const IMAGE_SIZE = 256;
const BITS_PER_BYTE = 8;

class App {
  constructor(image, canvas) {
    this.image = image;
    this.canvas = canvas;
  }

  run() {
    this.imageSplitter = new ImageSplitter(this.image, this.canvas);
    this.neuralNetwork = new NeuralNetwork(0.001, 800);
    this.neuralNetwork.train(this.imageSplitter.blocks);

    let rects = [];

    for (let bitMap of this.imageSplitter.blocks) {
      let rect = this.neuralNetwork.getRestoredValue(bitMap);
      rects.push(rect[0]);
    }

    const imageRestorer = new ImageRestorer(
      rects,
      RECT_SIZE,
      IMAGE_SIZE,
      IMAGE_SIZE,
      document.getElementById("canvas2")
    );
    const compressedImage = imageRestorer.restore();
    const compressionInfoSize =
      (compressedImage.length * BITS_PER_BYTE +
        INPUT_SIZE * BITS_PER_BYTE +
        this.canvas.width * this.canvas.height * BITS_PER_BYTE +
        RECT_SIZE * RECT_SIZE * BITS_PER_BYTE) *
      BITS_PER_BYTE;
    const imageSize = this.imageSplitter.blocks.reduce((acc, el) => acc + Math.pow(el.length, 2), 0);

    document.getElementById("koeff").append((imageSize * BITS_PER_BYTE) / compressionInfoSize);
  }
}

class ImageSplitter {
  bitMaps = [];

  constructor(image, canvas) {
    this.image = image;
    this.ctx = canvas.getContext("2d");
    this.blocks = [];

    this.splitImageOnBlocks();
  }

  splitImageOnBlocks() {
    for (let y = 0; y < this.image.height; y += RECT_SIZE) {
      for (let x = 0; x < this.image.width; x += RECT_SIZE) {
        const imageData = this.ctx.getImageData(x, y, RECT_SIZE, RECT_SIZE);

        const filteredRow = [];
        for (let i = 0; i < imageData.data.length; i += 4) {
          filteredRow.push((imageData.data[i] * 2) / 255 - 1); // Red
          filteredRow.push((imageData.data[i + 1] * 2) / 255 - 1); // Green
          filteredRow.push((imageData.data[i + 2] * 2) / 255 - 1); // Blue
        }

        this.blocks.push(filteredRow);
      }
    }
  }
}

class NeuralNetwork {
  firstLayerWeights = [];
  secondLayerWeights = [];

  constructor(learningRate, errorThreshold) {
    this.learningRate = learningRate;
    this.errorThreshold = errorThreshold;

    this.generateRandomWeights(INPUT_SIZE, 128);

    this.secondLayerWeights = MatrixHelper.transpose(this.firstLayerWeights);

    this.firstLayerWeights = this.normalize(this.firstLayerWeights);
    this.secondLayerWeights = this.normalize(this.secondLayerWeights);

    this.firstLayerWeights.map((vector) => this.normalizeVector(vector));
    this.secondLayerWeights.map((vector) => this.normalizeVector(vector));
  }

  train(vectorsOfColor) {
    let error = 999999;
    let errorsPerIter = 0;
    while (error > this.errorThreshold) {
      for (let vectorOfColor of vectorsOfColor) {
        let feedForwardedMattrixes = this.feedForward(vectorOfColor);
        errorsPerIter += this.backPropagate(vectorOfColor, feedForwardedMattrixes);
      }

      error = errorsPerIter;
      console.log(error);
      errorsPerIter = 0;
    }
  }

  getRestoredValue(vectorOfColor) {
    return this.feedForward(vectorOfColor).secondLayerMattrix;
  }

  feedForward(vectorOfColor) {
    vectorOfColor = [vectorOfColor];
    let firstLayerMattrix = MatrixHelper.multiply(vectorOfColor, this.firstLayerWeights);
    let secondLayerMattrix = MatrixHelper.multiply(firstLayerMattrix, this.secondLayerWeights);

    return {
      firstLayerMattrix,
      secondLayerMattrix,
    };
  }

  backPropagate(vectorOfColor, forwardedMatrices) {
    let { firstLayerMattrix, secondLayerMattrix } = forwardedMatrices;

    vectorOfColor = [vectorOfColor];

    let deltaLastLayer = MatrixHelper.subtract(secondLayerMattrix, vectorOfColor);

    this.secondLayerWeights = MatrixHelper.subtract(
      this.secondLayerWeights,
      MatrixHelper.transpose(
        MatrixHelper.multiplyByNumber(
          MatrixHelper.multiply(MatrixHelper.transpose(deltaLastLayer), firstLayerMattrix),
          this.learningRate
        )
      )
    );

    let deltaFirstLayer = MatrixHelper.multiply(deltaLastLayer, MatrixHelper.transpose(this.secondLayerWeights));

    this.firstLayerWeights = MatrixHelper.subtract(
      this.firstLayerWeights,
      MatrixHelper.multiplyByNumber(
        MatrixHelper.multiply(MatrixHelper.transpose(vectorOfColor), deltaFirstLayer),
        this.learningRate
      )
    );

    return deltaLastLayer.reduce((acc, row) => acc + row.reduce((sum, val) => sum + val ** 2, 0), 0);
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

  generateRandomWeights(inputSize, outputSize) {
    for (let i = 0; i < inputSize; i++) {
      this.firstLayerWeights.push(Array.from({ length: outputSize }, () => Math.random() * 2 - 1));
    }
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

class ImageRestorer {
  constructor(bitMaps, blockSize, width, height, canvas) {
    this.bitMaps = bitMaps;
    this.blockSize = blockSize;
    this.ctx = canvas.getContext("2d");
    this.width = width;
    this.height = height;
  }

  restore() {
    const imageData = this.ctx.createImageData(this.width, this.height);
    const data = imageData.data;

    let blockIndex = 0;
    for (let y = 0; y < this.height; y += this.blockSize) {
      for (let x = 0; x < this.width; x += this.blockSize) {
        const block = this.bitMaps[blockIndex];

        for (let i = 0; i < this.blockSize; i++) {
          for (let j = 0; j < this.blockSize; j++) {
            if (y + i < this.height && x + j < this.width) {
              const pixelIdx = (i * this.blockSize + j) * 3; // Индекс R в block
              const dataIdx = ((y + i) * this.width + (x + j)) * 4; // Индекс R в data

              data[dataIdx] = Math.round(((block[pixelIdx] + 1) * 255) / 2); // R
              data[dataIdx + 1] = Math.round(((block[pixelIdx + 1] + 1) * 255) / 2); // G
              data[dataIdx + 2] = Math.round(((block[pixelIdx + 2] + 1) * 255) / 2); // B
              data[dataIdx + 3] = 255; // Alpha
            }
          }
        }
        blockIndex++;
      }
    }

    this.ctx.putImageData(imageData, 0, 0);

    return data;
  }
}

class MatrixHelper {
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
