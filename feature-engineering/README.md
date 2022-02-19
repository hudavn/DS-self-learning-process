# Feature Engineering
Về kĩ thuật tạo đặc trưng chúng ta có 3 phương pháp chính:

- Trích lọc đặc trưng *(Feature Extraction)*: Không phải toàn bộ thông tin được cung cấp từ một biến dự báo hoàn toàn mang lại giá trị trong việc phân loại. Do đó chúng ta cần phải trích lọc những thông tin chính từ biến đó.

- Biến đổi đặc trưng *(Feature Transformation)*: Biến đổi dữ liệu gốc thành những dữ liệu phù hợp với mô hình nghiên cứu. Những biến này thường có tương quan cao hơn đối với biến mục tiêu và do đó giúp cải thiện độ chính xác của mô hình.

- Lựa chọn đặc trưng *(Feature Selection)*: Phương pháp này được áp dụng trong những trường hợp có rất nhiều dữ liệu mà chúng ta cần lựa chọn ra dữ liệu có ảnh hưởng lớn nhất đến sức mạnh phân loại của mô hình. 

## Dataset 
[Kaggle - Two Sigma Connect: Rental Listing Inquiries](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/overview)

Bài toán của chúng ta là cần dự báo mức độ tín nhiệm của một danh sách những người thuê mới. Chúng ta phân loại danh sách thành 3 cấp độ [‘low’, ‘medium’, ‘high’].

## Feature Extraction
Trong thực tế dữ liệu thường ở dạng thô, đến từ nhiều nguồn khác nhau như văn bản, các phiếu điều tra, các hệ thống lưu trữ, website, app, ,… Nên đòi hỏi người xây dựng mô hình phải thu thập và tổng hợp lại các nguồn dữ liệu có liên quan đến đề tài nghiên cứu. Dữ liệu sau đó phải được làm sạch và chuyển thành dạng có cấu trúc (structure data) để tiến hành xây dựng mô hình. Do đó chúng ta sẽ cần đến các kĩ thuật trích lọc đặc trưng để biến dữ liệu từ dạng thô sơ như text, word, các nhãn sang các biến số học có khả năng định lượng.

- Feature Extraction cho văn bản - **example01.py**:
    - Kĩ thuật mã hóa (tokenization) sẽ giúp ta thực hiện điều này. Mã hóa đơn giản là việc chúng ta chia đoạn văn thành các câu văn, các câu văn thành các từ. Trong mã hóa thì từ là đơn vị cơ sở. 
    
    - Chúng ta cần một bộ tokenizer có kích thước bằng toàn bộ các từ xuất hiện trong văn bản hoặc bằng toàn bộ các từ có trong từ điển. Một câu văn sẽ được biểu diễn bằng một sparse vector mà mỗi một phần tử đại diện cho một từ, giá trị của nó bằng 0 hoặc 1 tương ứng với từ không xuất hiện hoặc có xuất hiện. 

    - Các bộ tokernizer sẽ khác nhau cho mỗi một ngôn ngữ khác nhau. 

- Feature Extraction trong xử lý ảnh - **example02.py**: 
    - Xử lý ảnh là một lĩnh vực vừa dễ và vừa khó. Nó dễ bởi chúng ta có thể ứng dụng các mô hình pretrained mà không cần phải suy nhiều, nhưng nó cũng khó hơn bởi nếu bạn muốn xây dựng một mô hình cho riêng mình đòi hỏi bạn phải thực sự đào sâu vào nó.

    - Thông thường trong lĩnh vực computer vision chúng ta sẽ sử dụng mạng nơ-ron tích chập. Bạn không cần phải tìm ra kiến trúc và huấn luyện mạng từ đầu. Thay vào đó, có thể tải xuống một mạng hiện đại đã được pretrained với trọng số từ các nguồn đã được công bố. Các nhà khoa học dữ liệu thường thực hiện điều chỉnh để thích ứng với các mạng này theo nhu cầu của họ bằng cách “tách” các lớp kết nối đầy đủ (fully connected layers) cuối cùng của mạng, thêm các lớp mới được thiết kế cho một nhiệm vụ cụ thể, và sau đó đào tạo mạng trên dữ liệu mới. Nếu nhiệm vụ của bạn chỉ là vector hóa hình ảnh, bạn chỉ cần loại bỏ các lớp cuối cùng và sử dụng kết quả đầu ra từ các lớp trước đó.


## References
- [phamdinhkhanh.github.io](https://phamdinhkhanh.github.io/2019/01/07/Ky_thuat_feature_engineering.html)