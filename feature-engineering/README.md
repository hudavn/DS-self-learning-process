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

- Feature Extraction cho văn bản - **example01.py**


## References
- [phamdinhkhanh.github.io](https://phamdinhkhanh.github.io/2019/01/07/Ky_thuat_feature_engineering.html)