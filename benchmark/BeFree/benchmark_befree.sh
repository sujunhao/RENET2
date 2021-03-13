#!/bin/bash

####################################################################
#
#	 Copyright (C) 2017 Àlex Bravo and Laura I. Furlong, IBI group.
#
#	 BeFree is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    BeFree is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#	 How to cite BeFree:
#	 Bravo,À. et al. (2014) A knowledge-driven approach to extract disease-related biomarkers from the literature. Biomed Res. Int., 2014, 253128.
#	 Bravo,À. et al. (2015) Extraction of relations between genes and diseases from text and large-scale data analysis: implications for translational research. BMC Bioinformatics, 16, 55.
#
####################################################################

#######################
#     PARAMETERS      #
#######################


# Change these paths
DIRPATH="befree"
# DATABASE_NAME="medline"
# COLLECTION_NAME="NER"
# PMID_LIST_PATH=""

# Output Folder
OUTPUT_PATH="n_out/"

# File names
GENE_FILENAME="genes"
DISEASE_FILENAME="diseases"
COOC_FILENAME="cooc"

#RE Parameters
DEPENDENCIES="n"
MODEL_FILE_PATH="$DIRPATH/corpora/EUADR_JSRE_models/EUADR_target_disease_SL.model"

#######################
#      SCRIPTS        #
#######################
RE_COOC_PATH="$DIRPATH/src/re/BeFree_RE_JSRE_conversion.py"
JAR_CONVERSOR_PATH="$DIRPATH/src/re/BeFree2JSRE.jar"
JAR_PRED_PATH="$DIRPATH/src/re/BeFreeREPrediction.jar"
RE_SELECTION_PATH="$DIRPATH/src/re/BeFree_RE_pred_selection.py"
METRICS_PATH="calculate_metrics.py"

#######################
COOC_STEP1_PATH="$OUTPUT_PATH"
COOC_STEP1_PATH+="$COOC_FILENAME"
COOC_STEP1_PATH+="_list_jsre.befree"

OUTPUT_STEP1_PATH="$OUTPUT_PATH"
OUTPUT_STEP1_PATH+="$COOC_FILENAME"
OUTPUT_STEP1_PATH+="_list_jsre.jsre"

PRED_FILE_PATH="$OUTPUT_PATH"
PRED_FILE_PATH+="$COOC_FILENAME"
PRED_FILE_PATH+="_list_jsre.pred"

BEFREE_SPLIT_PLATH=$OUTPUT_PATH
BEFREE_SPLIT_PLATH+="befree_split"

COOC_FINAL_PATH="$OUTPUT_PATH"
COOC_FINAL_PATH+="$COOC_FILENAME"
COOC_FINAL_PATH+="_FINAL.befree"


# TOKEN_MODEL_PATH="/ibi/users/shared/BeFreeNER2016/src/re/JULIE_life-science-1.6.mod.gz"
# POS_MODEL_PATH="/ibi/users/shared/BeFreeNER2016/src/re/POSTagPennBioIE.bin.gz"
# POS_DICT_PATH="/ibi/users/shared/BeFreeNER2016/src/re/tagdictPennBioIE"
# STANFORD_MODEL_PATH="/ibi/users/shared/BeFreeNER2016/src/re/englishPCFG.ser.gz"
# LOG_CONF="/ibi/users/shared/BeFreeNER2016/src/re/log-config.txt"
# KERNEL_CONF="/ibi/users/shared/BeFreeNER2016/src/re/jsre-config.xml"
TOKEN_MODEL_PATH="$DIRPATH/src/re/JULIE_life-science-1.6.mod.gz"
POS_MODEL_PATH="$DIRPATH/src/re/POSTagPennBioIE.bin.gz"
POS_DICT_PATH="$DIRPATH/src/re/tagdictPennBioIE"
STANFORD_MODEL_PATH="$DIRPATH/src/re/englishPCFG.ser.gz"
LOG_CONF="$DIRPATH/src/re/log-config.txt"
KERNEL_CONF="$DIRPATH/src/re/jsre-config.xml"

#######################
#     EXECUTION       #
#######################

PYTHON="python"
JAVA="java"

# Co-occurrences
$PYTHON $RE_COOC_PATH $OUTPUT_PATH $GENE_FILENAME $DISEASE_FILENAME $COOC_FILENAME
echo "Co-occurrecences OK!"

# Convert 2 JSRE
$JAVA  -jar $JAR_CONVERSOR_PATH $COOC_STEP1_PATH $OUTPUT_STEP1_PATH $DEPENDENCIES $TOKEN_MODEL_PATH $POS_MODEL_PATH $POS_DICT_PATH $STANFORD_MODEL_PATH
echo "Conversion OK!"

# Prediction
split --lines=100000 --numeric-suffixes $OUTPUT_STEP1_PATH $BEFREE_SPLIT_PLATH
for filename in "$BEFREE_SPLIT_PLATH"*; do
    echo "Predicting... $filename"
    $JAVA  -jar $JAR_PRED_PATH -l $LOG_CONF -g $KERNEL_CONF $filename $MODEL_FILE_PATH "$filename".pred > /dev/null
done
#java -jar $JAR_PRED_PATH -l $LOG_CONF -g $KERNEL_CONF $OUTPUT_STEP1_PATH $MODEL_FILE_PATH $PRED_FILE_PATH
echo "Prediction OK!"

# Assoc. Selection
$PYTHON $RE_SELECTION_PATH $COOC_STEP1_PATH $BEFREE_SPLIT_PLATH $COOC_FINAL_PATH

rm "$BEFREE_SPLIT_PLATH"*
$PYTHON $METRICS_PATH
