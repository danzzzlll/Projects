## benchmarking_script:
 - b) Первая итерация warmup занимает больше всего времени - связано с тем что модель инициализируется, проходят первые холодные градиенты. На моем GPU один forward_pass для small модели занимает ~0.06c. Backward_pass длится для всех итераций, кроме первой длится дольше ~0.17c. Стандартное отклонение маленькое.
 - с) Для small моделей достаточно 1 итерации на warmup, следующие итерации уже идут стабильно по времени.

## nsys_profile:
 - a) В первой итерации больше всего времени уходит на forward_pass, в остальных больше времени тратится на backward.\
 - b) amper_sgemm_64x64_n.

## mixed_precision_accumulation:
 - a) the model parameters within the autocast context - fp32
      the output of the first feed-forward layer (ToyModel.fc1) - fp16
      the output of layer norm (ToyModel.ln) - fp32
      the model’s predicted logits - fp16
      the loss - fp 32
      and the model’s gradients - fp32

 - b) В bf16 типы такие же, как и в fp16. Для Layer Norm важна высокая точность для расчета mean, variance.
 - c) Модель в bf16 занимает в среднем на 30 процентов меньше, чем модель в fp32.

## memory_profiling:
 - a) Да, по пикам можно понять какой этап идет (forward, backward, optimizer).
 - b,c) Table below:

      | Context Length | Only Forward Pass | Full Training Step  | mixed_precision |
      |----------------|-------------------|---------------------|-----------------|
      | 128            | ~1.2 GB           | ~2.3 GB             | False           |
      | 256            | ~2.0 GB           | ~3.1 GB             | False           |
      | 512            | ~4.2 GB           | ~5.4 GB             | False           |
      | 512            | ~3.1 GB           | ~4.4 GB             | True            |
      |----------------|-------------------|---------------------|-----------------|   

     Да, mixed_precision действительно влияет на потребление GPU памяти, но для длинных послеовательностей не так сильно. Для context_length=512 удалось сэкономить по 1GB для forward и full pass.       
 - e) Самые большие allocations это матрицы (kernels) для attention.

## pytorch_attention

     Я зафиксировал model_dims = [16, 32, 64, 128]
          и seq_lens = [256, 1024, 4096, 8192, 16384]

     OOM я начинаю получать уже на этапе seq_len=16384 model_dim=16; Для d_model=128 и seq_len=8192 OOM не происходит, часть памяти докидывается с CPU и последовательность обрабатывается но очень медленно.
     Параметры: d_model * d_model + d_model = (128 * 128 + 128) * 4 ~ 65 MB
     Входы Q, K, V, Y: 4 * batch_size * seq_len * d_model * 4 (bytes) = 4 * 8 * 8192 * 128 * 4 
     <!-- Softmax(QK^T)V и выход Attention: 2 * batch_size * seq_len * d_model * 4 = 
     Активации после output_proj: batch_size * seq_len * d_model * 4 =  -->
     ----------------------------------
     whole_memory: (seq_len * seq_len + seq_len * seq_len + seq_len * d_model + 3 * seq_len * d_model) * batch_size
     (8192 * 8192 * 2 + 8192 * 128 + 3 * 8192 * 128) * 8 / 1024 / 1024 / 1024 ~ 4.1 GB

     Чтобы бороться OOM можно еще большую часть параметров перекидывать на cpu или использовать другие алгоритмы attention.


     




