COMPIlE_EXEC=mvNCCompile
MODEL_DIR=./data/models
MODEL0=rapidus-1
MODEL1=rapidus-hagl10

compile () {
	local PROTOTXT=${MODEL_DIR}/${1}.prototxt
	local GRAPH=${MODEL_DIR}/${1}.graph
	mvNCCompile ${PROTOTXT} -o ${GRAPH} -s 12
	if [ $? -eq 0 ]
	then
		echo 
		echo Successfully compiled model to movidius graph file ${GRAPH}
		echo
	else
		echo 
		echo Failed compiling model ${PROTOTXT}
		echo 
    fi
}

compile ${MODEL0}
compile ${MODEL1}
