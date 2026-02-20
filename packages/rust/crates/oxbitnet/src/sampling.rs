use rand::Rng;

/// Sample a token from logits with temperature, top-k, and repetition penalty.
pub fn sample_token(
    logits: &mut [f32],
    temperature: f32,
    top_k: usize,
    repeat_penalty: f32,
    recent_tokens: &[u32],
) -> u32 {
    let vocab_size = logits.len();

    // Repetition penalty (llama.cpp style)
    if repeat_penalty != 1.0 && !recent_tokens.is_empty() {
        for &token_id in recent_tokens {
            let idx = token_id as usize;
            if idx < vocab_size {
                if logits[idx] > 0.0 {
                    logits[idx] /= repeat_penalty;
                } else {
                    logits[idx] *= repeat_penalty;
                }
            }
        }
    }

    // Temperature
    if temperature != 1.0 {
        let inv_temp = 1.0 / temperature;
        for logit in logits.iter_mut() {
            *logit *= inv_temp;
        }
    }

    // Top-K via min-heap (O(V) instead of O(V log V) sort)
    if top_k > 0 && top_k < vocab_size {
        let mut heap: Vec<usize> = (0..top_k).collect();

        // Build initial min-heap
        for i in (0..(top_k / 2)).rev() {
            sift_down(&mut heap, i, top_k, logits);
        }

        // Process remaining
        for i in top_k..vocab_size {
            if logits[i] > logits[heap[0]] {
                heap[0] = i;
                sift_down(&mut heap, 0, top_k, logits);
            }
        }

        let threshold = logits[heap[0]];
        for i in 0..vocab_size {
            if logits[i] < threshold {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax + sample
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for logit in logits.iter_mut() {
        *logit = (*logit - max_val).exp();
        sum += *logit;
    }

    let mut rng = rand::rng();
    let r = rng.random::<f32>() * sum;
    let mut cumsum = 0.0f32;
    for (i, &logit) in logits.iter().enumerate() {
        cumsum += logit;
        if cumsum >= r {
            return i as u32;
        }
    }

    (vocab_size - 1) as u32
}

fn sift_down(heap: &mut [usize], mut i: usize, n: usize, values: &[f32]) {
    loop {
        let mut min = i;
        let l = 2 * i + 1;
        let r = 2 * i + 2;
        if l < n && values[heap[l]] < values[heap[min]] {
            min = l;
        }
        if r < n && values[heap[r]] < values[heap[min]] {
            min = r;
        }
        if min == i {
            break;
        }
        heap.swap(i, min);
        i = min;
    }
}
