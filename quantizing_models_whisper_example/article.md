Исходная статья доступна на 
https://medium.com/@daniel-klitzke/quantizing-openais-whisper-with-the-huggingface-optimum-library-30-faster-inference-64-36d9815190e0


* Author: Daniel Klitzke May 16, 2023
* Translation: Konstantin Zvyagin  Feb 25, 2024 

<h1>
Quantizing[Кванование] модели whisper OpenAI с помощью
библиотеки Huggingface Optimum Library от Huggingface, с результатом достижения ускорения на 30% и снижение требования по памяти на 64 %
</h1>


![График сокращения потребления памяти модели whisper ](https://raw.githubusercontent.com/kzvyagin/article_translations/main/quantizing_models_whisper_example/images/trend_1.png)

Как сэкономить 30 % времени работы inference нейронной сети и 64 % памяти при расшифровке аудио с помощью модели OpenAI Whisper, можно прочитав эту статью и запустив код приведенный в ней.

<h2>Введение</h2>

Ввиду развития и наличия большого количества готовых моделей нейросетей, возможности применять их к большому спектру задач, на первый взгляд кажется что нет необходимости создавать свою собственную модель и понимать как они устроены.  
Однако на практике это часто оказывается иллюзией! 

На практике можно столкнуться со следующим рабом проблем:

1. Модель имеет высокие требования к ресурсам и требует дорогостоящего оборудования, такого как
как наличие GPU определенной модели, чтобы обработка  происходила достаточно быстро.
2. Скорость выполнения инференса модели имеет слишком большую задержку для задачи.
3. Невозможно развернуть модель локально , необходимо пользоваться API поставщика модели (например, ChatGPT API) , что сильно влияет на задержку обработки.

В зависимости от варианта использования и предметной области будет актуален тот или иной пункт ихз списка выше. Например, предположим, что необходимо работать в среде энергоэффективных систем на базе микроконтроллеров или встраиваемых процессоров. В этом случае разработчик определенно почувствует боль с
самого начала. Если посмотреть на область приложений или сервисов где доступны стандартные ресурсы, то задача уже будет формулироваться скорее как предоставлении пользователю лучшего опыта или экономии затрат при масштабировании сервиса или приложения.  В любом случае актуальность задачи оптимизации готовой модели понятна.

Итак, каковы некоторые конкретные способы решения этих проблем:

1. Обучение совершенно новой модели в более узкой области 
2. Квантование [Quantization]
3. Обрезка [Pruning]
4. Приближение низкого ранга [ Low-rank approximation]
5. Дистилляция знаний / подходы учителя и ученика [Knowledge distillation / Teacher-Student approaches]
6. Использование фичей ускорения конкретного аппаратного модуля или GPU 

Как вы можете заметить, эти методы принципиально отличаются друг от друга, располагаясь как бы в разных направлениях оптимизации. Например, некоторые уменьшают размер модели по сравнению с оригиналом модели,другие требуют дополнительного этапа обучения, третьи просто оптимизации процесса inferenca[выводап] вывода без дополнительного обучения.

И поверьте мне, когда я говорю вам, что стоит дважды подумать о том, какой тип оптимизации выбрать и насколько это усложнит общий
конвейер обработки данных! 

Недавно я подумал, что было бы хорошей идеей улучшить модель whisper в OpenAI с помощью некоторой техники дистилляции, и установить что организация традиционного обучения значительно усложнила бы общий проект. Нужно будет не только написать цикл обучения, но и потенциально включить такие вещи, как отслеживание экспериментов, управление версиями данных и т.д.


Поэтому мой совет таков: если вы можете выбрать подход, при котором вы можете избежать каких-либо обучающих и ручных действий в архитектуре модели, то предпочитайте этот подход  всему остальному!


Ниже я приведу краткий пример того, как оптимизировать большую версию модели OpenAI Whisper взятую с  Hugging face Model Hub, экспортировав ее в формат ONNX и запустив модель в квантованной версии, используя возможности библиотеки Huggingface Optimum для квантования.


<h3>Шаг 1: Установка зависимостей</h3>

```python

!pip install -U optimum[exporters,onnxruntime] transformers torch

```

<h3>Шаг 2: Квантование модели</h3>

```python

    from pathlib import Path
    from optimum.onnxruntime import (AutoQuantizationConfig,
                                     ORTModelForSpeechSeq2Seq,ORTQuantizer
                                     )
    # Configure base model and save directory for compressed model
    model_id = "openai/whisper-large-v2"
    save_dir = "whisper-large"
    
    # Export model in ONNX
    model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
    model_dir = model.model_save_dir
    
    # Run quantization for all ONNX files of exported model
    onnx_models = list(Path(model_dir).glob("*.onnx"))
    print(onnx_models)
    quantizers = [ORTQuantizer.from_pretrained(model_dir, file_name=onnx_model) for onnx_model in onnx_models ]
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False )
    
    for quantizer in quantizers:
        # Apply dynamic quantization and save the resulting model
        quantizer.quantize(save_dir=save_dir, quantization_config=qconfig)

```

<h3>Шаг 3: Сравнение исходной и квантованной модели</h3>

```python


    from datetime import datetime
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor
    
    # Number of inferences for comparing timings
    num_inferences = 4
    save_dir = "whisper-large"
    inference_file = "test2.wav"
    
    # Create pipeline based on quantized ONNX model
    model = ORTModelForSpeechSeq2Seq.from_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    feature_extractor = AutoFeatureExtractor.from_pretrained(save_dir)
    cls_pipeline_onnx = pipeline("automatic-speech-recognition", model=model, tokenizer=tokinizer, feature_extractor=feature_extractor)
    
    # Create pipeline with original model as baseline
    cls_pipeline_original = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")

    # Measure inference of quantized model
    start_quantized = datetime.now()
    for i in range(num_inferences):
    cls_pipeline_onnx(inference_file)
    end_quantized = datetime.now()
    # Measure inference of original model
    start_original = datetime.now()
    for i in range(num_inferences):
    cls_pipeline_original(inference_file)
    end_original = datetime.now()
    original_inference_time = (end_original - start_original).total_seconds() / num_inferences
    print(f"Original inference time: {original_inference_time}")
    quantized_inference_time = (end_quantized - start_quantized).total_seconds() / num_inferences
    print(f"Quantized inference time: {quantized_inference_time}")

```

<h3>Выводы</h3>

При запуске квантованной модели на моем компьютере (на процессоре) ей требуется на 64% меньше
памяти и она работает более чем на 30% быстрее, обеспечивая сопоставимые результаты транскрибации.
Обратите внимание, что это основано на простом выполнении inference, без какой-либо
пакетной обработки или аналогичных оптимизаций. Тем не менее, я думаю, что это уже впечатляет, если
учесть, что при экономии времени на 30% и памяти на 64% вам не нужно было много делать работы самостоятельно, и смогли избежать настройки каких-либо циклов обучения или чего-либо подобного.
