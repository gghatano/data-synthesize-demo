# Data Synthesize Demo

A demonstration of data synthesis using SDV (Synthetic Data Vault) and Streamlit. This project showcases how to generate synthetic data while preserving or ignoring relationships between columns, using models like `GaussianCopulaSynthesizer` and `CTGANSynthesizer`.

## Features

- Generate synthetic data using SDV's advanced synthesizers.
- Compare models with or without inter-column relationships.
- Interactive web app powered by Streamlit.
- Easy deployment to Streamlit Cloud.

---

## Setup

Follow these steps to run the project locally.

### Prerequisites

- Python 3.12 or higher
- `pip` or `uv` for dependency management

### Clone the repository

```bash
git clone https://github.com/gghatano/data-synthesize-demo.git
cd data-synthesize-demo
```

### Install dependencies

#### Using `pip`:
```bash
pip install -r requirements.txt
```

#### Using `uv`:
```bash
uv install
```

---

## Running the App

After setting up the environment, start the Streamlit app:

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501` to interact with the app.

---

## Usage

1. Upload a CSV file containing your dataset.
2. Select a synthesis model:
   - **GaussianCopulaSynthesizer**: Preserves inter-column relationships.
   - **CTGANSynthesizer**: Handles complex, non-linear data distributions.
   - **BaseIndependentSampler**: Ignores inter-column relationships.
3. Specify the number of rows to generate.
4. Download the synthesized data as a CSV file.

---

## Deployment

This project is ready for deployment on Streamlit Cloud.

1. Push your changes to the `main` branch of your repository.
2. Visit [Streamlit Cloud](https://streamlit.io/cloud).
3. Link your GitHub repository and deploy the app.

---

## Project Structure

```
data-synthesize-demo/
├── app.py               # Streamlit app main script
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata and dependencies (if applicable)
├── README.md            # Project documentation
├── src/                 # Source code and utility functions
└── tests/               # Unit tests
```

---

## Dependencies

The key Python libraries used in this project include:

- [SDV](https://github.com/sdv-dev/SDV): For generating synthetic data.
- [Streamlit](https://streamlit.io/): For building the interactive web app.
- [Pandas](https://pandas.pydata.org/): For data manipulation and handling.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- The [SDV team](https://sdv.dev/) for their excellent library.
- The [Streamlit team](https://streamlit.io/) for their user-friendly app framework.
