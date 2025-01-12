// Лабораторная работа выполнена студентом 121701 Галуга М.В.
// Вариант 13 (Реализовать модель сети Джордана-Элмана с недовыпрямленной линейной функцией активации (Leaky RELU))
// Источник https://stackoverflow.com/questions/26836729/elman-network-not-stop
// Вспомогательные функции https://poe.com/

// --- Линейная функция активации: Leaky ReLU ---
function leakyRelu(x) {
  // Возвращает x, если x > 0; иначе возвращает 0.01 * x (небольшой утечкой)
  return x > 0 ? x : 0.01 * x;
}

// Производная функции активации Leaky ReLU
function leakyReluDerivative(x) {
  // Производная равна 1 для x > 0; иначе 0.01
  return x > 0 ? 1 : 0.01;
}

// --- Сеть Джордана-Элмана ---
class JordanElmanNetwork {
  constructor(inputSize, hiddenSize, outputSize, learningRate) {
    // Размер входного, скрытого и выходного слоев, а также скорость обучения
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.learningRate = learningRate;

    // Инициализация весов для входного -> скрытого слоя
    this.inputWeights = Array.from({ length: hiddenSize }, () =>
      Array.from({ length: inputSize }, () => Math.random() * 0.2 - 0.1)
    );

    // Инициализация весов для скрытого -> выходного слоя
    this.hiddenWeights = Array.from({ length: outputSize }, () =>
      Array.from({ length: hiddenSize }, () => Math.random() * 0.2 - 0.1)
    );

    // Инициализация весов контекстных нейронов
    this.contextWeights = Array.from({ length: hiddenSize }, () => Math.random() * 0.2 - 0.1);

    // Контекстные нейроны (начальные значения равны 0)
    this.context = Array.from({ length: hiddenSize }, () => 0);
  }

  // Прямой проход (forward pass)
  forward(input) {
    // Вычисляем значения скрытого слоя с учетом входов и контекстных нейронов
    this.hidden = this.inputWeights.map((weights, j) => {
      let sum = weights.reduce((acc, w, i) => acc + w * input[i], 0); // Сумма взвешенных входов
      sum += this.context[j] * this.contextWeights[j]; // Добавляем взвешенные контекстные значения
      return leakyRelu(sum); // Применяем функцию активации
    });

    // Вычисляем значения выходного слоя
    this.output = this.hiddenWeights.map((weights) =>
      leakyRelu(weights.reduce((acc, w, j) => acc + w * this.hidden[j], 0))
    );

    return this.output; // Возвращаем выходные значения
  }

  // Обратное распространение ошибки (backpropagation)
  backward(input, target) {
    // Вычисляем ошибки на выходном слое
    const outputErrors = this.output.map((o, i) => (o - target[i]) * leakyReluDerivative(o));

    // Вычисляем ошибки на скрытом слое
    const hiddenErrors = this.hidden.map(
      (h, j) =>
        this.hiddenWeights.reduce((acc, weights, i) => acc + weights[j] * outputErrors[i], 0) * leakyReluDerivative(h)
    );

    // Обновление весов скрытого -> выходного слоя
    this.hiddenWeights = this.hiddenWeights.map((weights, i) =>
      weights.map((w, j) => w - this.learningRate * outputErrors[i] * this.hidden[j])
    );

    // Обновление весов вход -> скрытого слоя
    this.inputWeights = this.inputWeights.map((weights, j) =>
      weights.map((w, i) => w - this.learningRate * hiddenErrors[j] * input[i])
    );

    // Обновление весов контекстных нейронов
    this.contextWeights = this.contextWeights.map((w, j) => w - this.learningRate * hiddenErrors[j] * this.context[j]);

    // Обновление значений контекстных нейронов (текущие скрытые значения становятся контекстом)
    this.context = [...this.hidden];
  }

  // Обучение сети на одном примере (input -> target)
  train(input, target) {
    this.forward(input); // Прямой проход
    this.backward(input, target); // Обратное распространение
  }
}

// --- Демонстрация работы сети ---
const fibonacci = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]; // Последовательность Фибоначчи
const repeating = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // Повторяющаяся последовательность

// Преобразование последовательностей в обучающие данные (input/target пары)
function prepareData(sequence) {
  const inputs = [];
  const targets = [];
  for (let i = 0; i < sequence.length - 1; i++) {
    inputs.push([sequence[i]]); // Текущий элемент как вход
    targets.push([sequence[i + 1]]); // Следующий элемент как целевое значение
  }
  return { inputs, targets };
}

// Инициализация сети
const network = new JordanElmanNetwork(1, 25, 1, 0.01); // 1 вход, 25 скрытых нейронов, 1 выход

// Обучение сети на числах Фибоначчи
const fibData = prepareData(fibonacci);
for (let epoch = 0; epoch < 1000; epoch++) {
  // 1000 эпох обучения
  fibData.inputs.forEach((input, i) => {
    network.train(input, fibData.targets[i]);
  });
}

// Тестирование сети на числах Фибоначчи
console.log("Числа Фибоначчи:");
fibData.inputs.forEach((input) => {
  console.log(`Input: ${input}, Predicted the next: ${network.forward(input)}`); // Вывод входов и предсказаний
});

// Обучение сети на повторяющемся ряде
const repData = prepareData(repeating);
for (let epoch = 0; epoch < 1000; epoch++) {
  // 1000 эпох обучения
  repData.inputs.forEach((input, i) => {
    network.train(input, repData.targets[i]);
  });
}

// Тестирование сети на повторяющемся ряде
console.log("Арифметическая прогрессия:");
repData.inputs.forEach((input) => {
  console.log(`Input: ${input}, Predicted the next: ${network.forward(input)}`); // Вывод входов и предсказаний
});
