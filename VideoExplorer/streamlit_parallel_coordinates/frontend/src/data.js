function generateFakeData(){
    const columnNames = ["#1", "#2", "#3", "#4", "#5", "#6"]
    let dimensions = [

    ]
    for (let columnName of columnNames){
        dimensions.push({
            "key": columnName,
            "type": "Number",
        })
    }
    let data = []
    for (let i = 0; i < 100; i++){
        let obj = {}
        for (let columnName of columnNames){
            obj[columnName] = Math.random() * 180
        }
        data.push({
            ...obj,
            file_id: i.toString()
        })
    }
    return {data, dimensions}
}
export {generateFakeData}