# design4emergency


## Setup

Clone repo:
```
git clone git@github.com:tezzutezzu/design4emergency.git
```

Enter repo directory:
```
cd design4emergency
```

Create virtualenv:
```
viertualenv -p python3 venv
```

Enter in the virtualenv:
```
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Command:
```
python text_analysis.py path/to/dataset column_name
```

Example:
```
python text_analysis.py "/home/piero/data/DfE_Dataset_Corriere_1600_Danilo - Form Responses 1.tsv" "Cosa ti fa più paura?"
```

For more parameters check:
```
python text_analysis.py  -h
```
