## Data pipeline

### 1. DVC

The `archive.zip` file can be downloaded from Kaggle here: \
https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020\
Make sure it's placed in on this filepath:\
`src/anime_recommender/data/archive.zip`.

Alternatively, using `$uv run dvc pull`, DVC fetches the zip file instantly, since it's already pushed in a personal GoogleDrive using a specific version of it.

However a GoogleDrive Service Account JSON file is needed, in order to achieve this. It's under `.dvc/config.local` since `--local` tag was used in the command:\
`$uv run dvc remote modify --local storage \ gdrive_service_account_json_file_path <path-to-json-file>`.


It's recommended to simply *download* the file since it will be easier, although the versions might differ since these datasets tend to get often updated. Also, *uploading* it to AWS takes several minutes.

<hr>

### 2. Raw

This project has a script pointing to the `__main__.data` function, with name **ars-data**, which is a `click.group`.

In order to get the raw data in CSV format, simply use:\
`$ars-data load -o <name-of-raw-csv-file.csv>`\
The default value of the CSV file is: "anime-genre.csv".\
This file will be used later, in the prediction stage.\
An extra file is created: `dimension.txt`,\
which will be used in the Training Job, so ignore this for now.

To get the splitted training and testing data in CSV format, use:\
`$ars-data split --ratio <train-split-ratio> --seed <your-seed>`

<hr>

### 3. RecordIO - SVM

Factorization Machines need:

- for **Input**: only `recordIO-protobuf` format with `Float32` tensors.

- for **Inference**: `application/json` or `x-recordio-protobuf` formats.


To create training and testing recordio files, you can use:\
`$ars-data recordio-format --ratio <train-split-ratio> --seed <your-seed>`

**SIDEBAR**
1) This process takes a while since the Compressed Sparse Matrix is huge.
2) Keep the ratio and seed consistent through out all the steps.

Moreover, these extra `svmlight` files need to be created, using:\
`$ars-data svm-format --ratio <train-split-ratio> --seed <your-seed>`

<hr>

### 4. Lookup files

We'll create two lookup files:

- Categorical AnimeID and corresponding AnimeIndex in One-hot Encoded table
- Categorical UserID and corresponding UserIndex in One-hot Encoded table

Both files should be saved in `svmlight` format. Use:\
`$ars-data lookup-files --ratio <train-split-ratio> --seed <your-seed>`

Again, keep the train split ratio and seed consistent across all steps.
