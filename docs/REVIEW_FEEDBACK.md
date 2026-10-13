## 

## Submission Results

Submission Passed  
Congratulations\! Your submission has been reviewed and passed.

## Congratulation

Great work completing the project\! The submitted notebook provides a complete and verifiable end-to-end Transformer sentiment-analysis workflow.

The IMDB datasets load with the required dimensions, the exploratory analysis includes descriptive statistics and two distinct properly labelled visualizations, the custom IMDBDataset passes all supplied assertions, and the DemoGPT classifier correctly performs sequence-level mean pooling and produces the required **torch.Size(\[32, 2\])** output.

The reusable accuracy function is implemented correctly, and all three training epochs complete successfully. Validation accuracy improves from **71.08%** to **78.24%**, while the final held-out test evaluation reaches **76.12%**, exceeding the project target of 75%. The conclusion also contains the required summary and multiple meaningful key takeaways.

The saved sentiment\_transformer.pt checkpoint and the additional inference helper are useful extensions because they demonstrate how the trained model can be reused outside the training loop. A strong next step would be to preserve the best validation checkpoint automatically and compare this custom Transformer with a pretrained Transformer fine-tuned for sequence classification.

### Load, Explore, and Prepare Data

### Requirements to Pass

✅ A helper function has been used to load the dataset

✅ The loaded dataframes have the correct dimension as given in the assert statements.

### Feedback

The dataset-loading workflow is implemented correctly. You created a reusable load\_dataset() helper, used it for all four positive/negative IMDB directories, and built the expected training and test DataFrames.

The supplied assertions execute successfully, confirming that both train\_df and test\_df contain the required **25,000 rows and 2 columns**.

A useful reproducibility improvement would be to sort filenames before reading them so the loading order remains deterministic across environments.

for filename in **sorted**(os.**listdir**(folder)):

    ...

### Helpful resource

* [**Python sorted() documentation**](https://docs.python.org/3/library/functions.html#sorted)  
* [**(opens in a new tab)**](https://docs.python.org/3/library/functions.html#sorted)  
*   
* A helper function has been used to load the dataset  
* The loaded dataframes have the correct dimension as given in the assert statements.

### Requirements to Pass

✅ Descriptive statistics associated to the dataset have been calculated

✅ At least 2 key metrics of the datasets have been visualized with proper labelling on the graphs

### Feedback

The exploratory analysis fully satisfies this rubric.

You inspect the dataset with info(), describe(), and class-count analysis, and you also calculate character- and word-length statistics. The saved output shows approximately **233.79 words per review on average**, a median of **174 words**, and a maximum of **2,470 words**.

You also render two distinct and properly labelled visualizations:

* **Distribution of Sentiment Labels in Training Set**  
* **Distribution of review word counts by Sentiment**

Both graphs include clear titles, x-axis labels, and y-axis labels. The second visualization also marks the 128-token context length, which connects the EDA to the later model configuration.

A useful extension would be to summarize review-length statistics separately by sentiment class.

train\_df.**groupby**("label")\["word\_count"\].**describe**()

### Helpful resource

* [**Pandas DataFrameGroupBy.describe() documentation**](https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.core.groupby.DataFrameGroupBy.describe.html)  
* [**(opens in a new tab)**](https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.core.groupby.DataFrameGroupBy.describe.html)  
*   
* Descriptive statistics associated to the dataset have been calculated  
* At least 2 key metrics of the datasets have been visualized with proper labelling on the graphs

Reviewer Note

### Requirements to Pass

✅ The \_\_init\_\_() , \_\_len\_\_() , and \_\_getitiem\_\_() functions have been implemented according to the requirements.

✅ IMDBDataset objects instantiated for training data, validation data, and test data have the right lengths and data types as given in the assert statements.

### Feedback

The custom IMDBDataset is implemented correctly. The class defines the required initialization, length, and item-retrieval behavior and tokenizes each review with truncation, fixed-length padding, the configured maximum sequence length, and PyTorch tensor output.

Applying .squeeze(0) correctly produces the expected one-dimensional token-ID tensor.

The supplied assertions execute successfully for:

* **22,500 training samples**  
* **2,500 validation samples**  
* **25,000 test samples**

The assertions also verify the expected input tensor type, integer-like label type, and sequence length.

For slightly cleaner positional access, you could retrieve the selected row once before extracting its fields.

**row** \= self.**data**.iloc\[idx\]  
**text** \= str(**row**\["review"\])

label \= **int**(**row**\["label"\])

### Helpful resource

* [**Pandas DataFrame.iloc documentation**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html)  
* [**(opens in a new tab)**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html)  
*   
* The \_\_init\_\_() , \_\_len\_\_() , and \_\_getitiem\_\_() functions have been implemented according to the requirements.  
* IMDBDataset objects instantiated for training data, validation data, and test data have the right lengths and data types as given in the assert statements.

### Model Definition and Training

### Requirements to Pass

✅ The function definition of the calculate\_accuracy() function has been completed according to the requirements.

### Feedback

The calculate\_accuracy() function is implemented correctly. It switches the model to evaluation mode with model.eval(), disables gradient computation with torch.no\_grad(), iterates over the supplied data\_loader, moves inputs and labels to the selected device, obtains predictions with torch.argmax(), and accumulates accuracy over the complete dataset.

The function is executed before training and reports an initial validation accuracy of **49.84%**, which is consistent with chance-level performance for a randomly initialized binary classifier.

A useful extension would be to retain predictions and labels during evaluation so additional metrics can be calculated without repeating inference.

all\_predictions.**extend**(predictions.**cpu**().**tolist**())

all\_labels.**extend**(labels.**cpu**().**tolist**())

### Helpful resource

* [**scikit-learn classification metrics**](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)  
* [**(opens in a new tab)**](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)  
* 

The function definition of the calculate\_accuracy() function has been completed according to the requirements.

### Requirements to Pass

✅ The training loop for the transformer model has been completed according to the requirement.

✅ The model has been trained without errors.

### Feedback

The training loop contains all required optimization steps: forward pass, cross-entropy loss calculation, optimizer.zero\_grad(), backward propagation, and optimizer update.

All three epochs complete successfully. The saved validation results are:

* Epoch 1: **71.08%**  
* Epoch 2: **75.80%**  
* Epoch 3: **78.24%**

The logged training loss also decreases over the run, which is consistent with successful learning.

A useful improvement would be to preserve the checkpoint with the best validation accuracy during training so the strongest model can be restored later if more epochs are added.

if val\_accuracy \> best\_val\_accuracy:  
    best\_val\_accuracy \= val\_accuracy

    torch.**save**(model.state\_**dict**(), "best\_model.pt")

### Helpful resource

* [**PyTorch — Saving and Loading Models**](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)  
* [**(opens in a new tab)**](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)  
*   
* The training loop for the transformer model has been completed according to the requirement.  
* The model has been trained without errors.

### Requirements to Pass

✅ The \_\_init\_\_() and \_\_forward\_\_() functions of the DemoGPT class have been implemented according to the requirements.

✅ Output from the DemoGPT class has the correct shape for the given dummy input as given in the assert statement.

### Feedback

The DemoGPT architecture is correctly customized for binary sentiment classification. The model defines token embeddings, positional embeddings, four Transformer blocks, layer normalization, and a two-class linear classification head.

The forward pass correctly performs mean pooling over the sequence dimension:

pooled \= torch.mean(x, dim=1)

logits \= self.classifier(pooled)

This converts the (batch, sequence, embedding) representation into one vector per review before classification.

The executed dummy-input check reports:

DemoGPT output shape: torch.Size(\[32, 2\])

and the supplied output-shape assertions execute successfully.

* The \_\_init\_\_() and \_\_forward\_\_() functions of the DemoGPT class have been implemented according to the requirements.  
* Output from the DemoGPT class has the correct shape for the given dummy input as given in the assert statement.

### Industry Best Practices

### Requirements to Pass

✅ Accuracy on the test dataset has been calculated.

✅ The test accuracy is greater than 75%.

### Feedback

The final evaluation correctly uses the held-out test DataLoader:

test\_accuracy \= calculate\_accuracy(model, test\_loader, device)

**print**(f"Final Test Accuracy: {test\_accuracy:.2f}%")

The saved output reports:

Final Test Accuracy: 76.12%

This exceeds the required 75% threshold.

You also save the trained model checkpoint and successfully run an additional single-review inference helper, which demonstrates that the trained model can be reused after evaluation.

A strong next evaluation step would be to inspect class-specific precision, recall, and F1-score so you can determine whether positive and negative reviews exhibit different error patterns.

### Helpful resource

* [**scikit-learn classification\_report documentation**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)  
* [**(opens in a new tab)**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)  
*   
* Accuracy on the test dataset has been calculated.  
* The test accuracy is greater than 75%.

### Requirements to Pass

✅ Project results have been summarized

✅ At least 2 key takeaways have been listed

### Feedback

The conclusion summarizes the main experiment, including the near-random baseline, the three-epoch training process, decreasing loss, and the final held-out test performance.

You also provide at least two meaningful key takeaways covering:

* adapting a Transformer to sequence classification using mean pooling and a linear classification layer, and  
* the role of BERT subword tokenization together with fixed-length truncation and padding.

The saved notebook reports **49.84% initial validation accuracy**, **78.24% final validation accuracy**, and **76.12% test accuracy**. The conclusion currently describes these values approximately, which is acceptable, but reporting the exact executed values would make the final report more precise.

Initial validation accuracy: 49.84%  
Final validation accuracy: 78.24%

Final test accuracy: 76.12%

### Helpful resource

* [**Model Cards for Model Reporting**](https://arxiv.org/abs/1810.03993)  
* [**(opens in a new tab)**](https://arxiv.org/abs/1810.03993)  
*   
* Project results have been summarized  
* At least 2 key takeaways have been listed

