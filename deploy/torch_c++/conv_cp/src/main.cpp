#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "convnet.h"

int main(){
	std::cout << "Convnet !!" << "\n\n";

	//Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "Training on gpu" : "Training on cpu") << "\n";

	//Hyper parameters
	const int64_t num_classes = 10;
	const int64_t batch_size = 64;
	const size_t num_epoch = 20;
	const double learning_rate = 0.001;

	const std::string mnist_path = "./data";

	//MNIST dataset
	auto train_dataset = torch::data::datasets::MNIST(mnist_path)
			.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
			.map(torch::data::transforms::Stack<>());

	//Numer of samples of train dataset 
	auto train_samples = train_dataset.size().value();

	//Test dataset
	auto test_dataset = torch::data::datasets::MNIST(mnist_path, torch::data::datasets::MNIST::Mode::kTest)
			.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
			.map(torch::data::transforms::Stack<>());

	//Number od samples of test dataset
	auto test_samples = test_dataset.size().value();

	//Data loader
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(train_dataset), batch_size);

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(test_dataset), batch_size);

	//Model
	ConvNet model(num_classes);
	model->to(device);

	//Optimizer
	torch::optim::SGD optimizer(model->parameters(), /*lr=*/0.01);

	//Set output precision
	std::cout << std::fixed << std::setprecision(4);
	
	std::cout << "start training..." << "\n";

	//Initialize best accuracy
	auto best_accuracy = 0.0;

	//Train model
	for(size_t epoch = 0; epoch < num_epoch; epoch++){
		//Train
		size_t batch_index = 0;
		model->train();
		for(auto &batch: *train_loader){
			auto data = batch.data.to(device);
			auto target = batch.target.to(device);
			//Forward
			optimizer.zero_grad();
			torch::Tensor prediction = model->forward(data);
			auto loss = torch::nn::functional::cross_entropy(prediction, target);
			//Backward
			loss.backward();
			optimizer.step();

			if(++batch_index % 100 == 0){
				std::cout << "Epoch: " << epoch + 1 << " |Batch: " << batch_index << " |Loss: " << loss.item<double>() << "\n";
			}
		}

		//Test
		model->eval();
		size_t num_correct = 0;
		for(auto &batch: *test_loader){
			auto data = batch.data.to(device);
			auto target = batch.target.to(device);
			//Forward
			torch::Tensor prediction = model->forward(data);
			prediction = prediction.argmax(1);
			//Compute number of correct
			num_correct += prediction.eq(target).sum().item<int64_t>();
		}

		auto accuracy = static_cast<double>(num_correct) / test_samples;
		std::cout << "Accuracy of epoch " << epoch + 1 << " is: " << accuracy << "\n";

		//Save model
		torch::save(model, "./models/last.pt");

		if(accuracy > best_accuracy){
			torch::save(model, "./models/best.pt");
			best_accuracy = accuracy;
		}
	}
}
