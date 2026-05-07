# Lab 21 - Báo cáo đánh giá

**Học viên**: Hoàng Quốc Chung - 2A202600070  
**Ngày nộp**: 2026-05-07  
**Hình thức nộp**: Option A - Lightweight ZIP  

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Phương pháp huấn luyện**: QLoRA 4-bit kết hợp LoRA adapters
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`
- **Số lượng mẫu sử dụng**: 200 samples
- **Chia train/eval**: 180 samples train, 20 samples eval
- **Định dạng dữ liệu**: Alpaca-style instruction tuning với `instruction`, `input`, và `output`
- **GPU**: Tesla T4, khoảng 16 GB VRAM
- **Phân bố độ dài token**: min = 25, p50 = 227, p95 = 562, p99 = 704, max = 738
- **max_seq_length**: 1024
- **LoRA target modules**: `q_proj`, `v_proj`
- **LoRA dropout**: 0
- **Gradient checkpointing**: bật với Unsloth
- **Số epoch**: 3
- **Learning rate**: `2e-4`
- **LR scheduler**: cosine
- **Effective batch size**: 8
- **Optimizer**: `adamw_8bit`
- **Tổng thời gian train**: 12.08 phút
- **Chi phí ước tính**: khoảng $0.07, giả định T4 có giá $0.35/giờ

## 2. Kết quả thí nghiệm Rank

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 1,843,200 | 4.01 phút | 11.21 GB | 1.5577 | 4.7479 |
| 16 | 32 | 3,686,400 | 4.12 phút | 10.61 GB | 1.5161 | 4.5544 |
| 64 | 128 | 14,745,600 | 3.95 phút | 12.00 GB | 1.4768 | 4.3790 |

Kết quả cho thấy khi tăng rank LoRA, eval loss và perplexity đều được cải thiện trên tập eval. Rank 64 đạt perplexity thấp nhất, tức là tốt nhất nếu chỉ xét metric định lượng. Tuy nhiên, rank 64 có 14.7 triệu tham số cần train, lớn gấp 4 lần rank 16 và gấp 8 lần rank 8. Rank 16 là cấu hình cân bằng hơn vì perplexity tốt hơn rank 8 khá rõ, nhưng kích thước adapter vẫn nhỏ hơn nhiều so với rank 64.

## 3. Phân tích Loss Curve

Loss curve trong quá trình training có xu hướng giảm, cho thấy adapter học được pattern từ dataset instruction tiếng Việt. Notebook không bật evaluation giữa các step để tránh lỗi out-of-memory trên GPU T4, vì vậy phân tích overfitting chủ yếu dựa trên eval loss cuối cùng và các ví dụ qualitative.

Không có dấu hiệu định lượng rõ ràng về overfitting nghiêm trọng, vì eval loss cuối cùng ở cả ba rank đều hợp lý và giảm khi rank tăng. Tuy nhiên, các ví dụ sinh câu trả lời cho thấy model fine-tuned vẫn có thể sai kiến thức, đặc biệt với các khái niệm kỹ thuật như LoRA và QLoRA. Điều này cho thấy model có thể đang học style và pattern của dataset nhỏ, nhưng chưa đảm bảo cải thiện độ chính xác về mặt kiến thức. Vì chỉ sử dụng 200 samples, thí nghiệm này nên được hiểu là một bài so sánh rank trong lab, không phải một lần fine-tuning đủ chất lượng để đưa vào production.

## 4. So sánh Qualitative

### Ví dụ 1

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

**Base model**: Base model giải thích machine learning là một nhánh của trí tuệ nhân tạo, nói về việc máy học từ dữ liệu và dùng mô hình để dự đoán hoặc hành động.

**Fine-tuned r=16**: Model fine-tuned giải thích machine learning là một lĩnh vực công nghệ máy tính dựa trên việc học từ dữ liệu, cải thiện dự đoán, và sử dụng thuật toán cùng mô hình học máy.

**Nhận xét**: Cả hai câu trả lời đều chấp nhận được. Câu trả lời của model fine-tuned trực tiếp hơn và phù hợp với kiểu giải thích giáo dục, nhưng vẫn còn khá tổng quát. Đây là một cải thiện nhẹ, không phải khác biệt lớn.

### Ví dụ 2

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

**Base model**: Base model đưa ra cách viết hàm Fibonacci và bắt đầu xử lý trường hợp số nguyên dương, nhưng cách trình bày nghiêng về đệ quy và output bị cắt ngắn.

**Fine-tuned r=16**: Model fine-tuned đưa ra cách cài đặt dùng biến `a, b = 0, 1` và vòng lặp để tính Fibonacci.

**Nhận xét**: Fine-tuned model tốt hơn trong ví dụ này vì cách làm bằng vòng lặp thực tế và hiệu quả hơn đệ quy naive. Đây là một case cải thiện rõ của model sau fine-tuning.

### Ví dụ 3

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

**Base model**: Base model đưa ra các nguyên tắc rộng như thân thiện với người dùng, sắp xếp bố cục, màu sắc và font chữ.

**Fine-tuned r=16**: Model fine-tuned liệt kê các ý như chuyển đổi, thích ứng, đơn giản và tương thích.

**Nhận xét**: Câu trả lời fine-tuned ngắn gọn hơn, nhưng một số cách dùng từ chưa tự nhiên. Base model dài hơn nhưng có phần gần với cách diễn đạt thông thường hơn về UI/UX. Ví dụ này là kết quả lẫn lộn, không phải cải thiện hoàn toàn.

### Ví dụ 4

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

**Base model**: Base model mở rộng đúng LoRA là Low-Rank Adaptation và QLoRA là Quantized LoRA, nhưng phần giải thích còn mơ hồ.

**Fine-tuned r=16**: Model fine-tuned giải thích sai LoRA thành "Layer-wise Adaptive Regularization Optimization" và xem nó như một kỹ thuật regularization.

**Nhận xét**: Đây là một failure case. Model fine-tuned trả lời tự tin nhưng sai định nghĩa. Ví dụ này cho thấy perplexity thấp hơn không đồng nghĩa với việc model luôn đúng về kiến thức. Với các câu hỏi kỹ thuật, cần dataset chất lượng hơn hoặc kết hợp RAG để giảm hallucination.

### Ví dụ 5

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

**Base model**: Base model giải thích đây là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học, nhưng nội dung còn chung chung.

**Fine-tuned r=16**: Model fine-tuned giải thích prompt engineering là xây dựng prompt, RAG là dùng ngữ cảnh truy xuất, và fine-tuning là điều chỉnh model bằng dữ liệu huấn luyện.

**Nhận xét**: Fine-tuned model rõ ràng hơn và gần với nội dung bài học hơn. Câu trả lời tách bạch ba kỹ thuật tốt hơn, nên đây là một ví dụ cải thiện có ý nghĩa.

## 5. Kết luận về Rank Trade-off

Trong thí nghiệm này, rank 64 đạt eval loss và perplexity tốt nhất. Perplexity giảm từ 4.7479 ở rank 8 xuống 4.5544 ở rank 16 và 4.3790 ở rank 64. Điều này cho thấy khi tăng rank, adapter có nhiều capacity hơn để học pattern của dataset instruction tiếng Việt. Tuy nhiên, mức cải thiện từ rank 16 lên rank 64 không lớn nếu so với chi phí về số tham số. Rank 64 có 14,745,600 trainable parameters, trong khi rank 16 chỉ có 3,686,400. Nghĩa là rank 64 lớn gấp 4 lần rank 16, nhưng perplexity chỉ cải thiện khoảng 0.175. Với dataset nhỏ 200 samples và môi trường GPU T4, rank 16 là lựa chọn thực tế hơn vì cân bằng tốt giữa chất lượng, kích thước adapter và tài nguyên tính toán. Nếu mục tiêu chỉ là đạt perplexity thấp nhất thì rank 64 tốt hơn, nhưng nếu cần deploy hoặc nộp adapter gọn nhẹ thì rank 16 hợp lý hơn. Ngoài ra, phần qualitative cho thấy model vẫn có thể hallucinate, nên không nên đánh giá chất lượng chỉ dựa vào perplexity.

## 6. Những điều tôi học được

- LoRA giúp fine-tune LLM với chi phí thấp hơn vì chỉ train adapter nhỏ thay vì cập nhật toàn bộ base model.
- QLoRA giúp load base model ở 4-bit, nhờ đó có thể fine-tune model 3B trên GPU T4.
- Tăng rank có thể cải thiện perplexity, nhưng hiệu quả biên giảm khi rank quá lớn so với kích thước dataset.
- Perplexity là metric hữu ích, nhưng vẫn cần đọc output qualitative vì model có thể trả lời sai một cách rất tự tin.

