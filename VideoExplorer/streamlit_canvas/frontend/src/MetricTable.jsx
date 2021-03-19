import React, {useEffect, useState} from 'react';
import {useTable} from 'react-table';
import './App.css';
import CssBaseline from '@material-ui/core/CssBaseline'
import MaUTable from '@material-ui/core/Table'
import TableBody from '@material-ui/core/TableBody'
import TableCell from '@material-ui/core/TableCell'
import TableHead from '@material-ui/core/TableHead'
import TableRow from '@material-ui/core/TableRow'
import Checkbox from '@material-ui/core/Checkbox';
import {ThemeProvider} from '@material-ui/core/styles';
import { createMuiTheme } from '@material-ui/core/styles';
import TextField from '@material-ui/core/TextField';

const theme = createMuiTheme({
    typography: {
        // 中文字符和日文字符通常比较大，
        // 所以选用一个略小的 fontsize 会比较合适。
        fontSize: 12,
    },
});

const EditableCell = ({
                          value: initialValue,
                          row: {index},
                          column: {id},
                          updateMyData, // This is a custom function that we supplied to our table instance
                      }) => {
    // We need to keep and update the state of the cell normally
    const [value, setValue] = React.useState(initialValue)

    const onChange = e => {
        setValue(e.target.value)
    }

    // We'll only update the external data when the input is blurred
    const onBlur = () => {
        updateMyData(index, id, value)
    }

    // If the initialValue is changed external, sync it up with our state
    React.useEffect(() => {
        setValue(initialValue)
    }, [initialValue])
    let styles = {
        width: 80
    }
    return <TextField style={styles} value={value} onChange={onChange} onBlur={onBlur}/>
    // return <input value={value} onChange={onChange} onBlur={onBlur}/>

}
// const defaultColumn = {
//     Cell: EditableCell,
// }

function MetricTable({metricData, setData}) {

    // const data = React.useMemo(
    //     () => metricData,
    //     []
    // );
    let data = metricData;
    const updateMyData = (rowIndex, columnId, value) => {
        // We also turn on the flag to not reset the page
        // setData(old =>
        //     old.map((row, index) => {
        //         if (index === rowIndex) {
        //             return {
        //                 ...old[rowIndex],
        //                 [columnId]: value,
        //             }
        //         }
        //         return row
        //     })
        // )
    }
    const styles = theme => ({
        root: {
            width: '100%',
            marginTop: theme.spacing.unit * 3,
            overflowX: 'auto',
        },
        table: {
            minWidth: 700
        },
        tablecell: {
            fontSize: '40pt'
        }
    });

    const columns = React.useMemo(
        () => [
            {
                Header: 'Name',
                accessor: 'name', // accessor is the "key" in the data,
                Cell: EditableCell
            },
            {
                Header: 'Id',
                accessor: 'id',
            },
            {
                Header: "Description",
                accessor: "description"
            },
            {
                Header: "Visible",
                accessor: 'visibility',
                type: "checkbox",
            }
        ],
        []
    )

    const tableInstance = useTable({columns, data, updateMyData})
    const {
        getTableProps,
        getTableBodyProps,
        headerGroups,
        rows,
        prepareRow,
    } = tableInstance

    return (
        <div>
            <ThemeProvider theme={theme}>

                <CssBaseline/>

                <MaUTable {...getTableProps()}>
                    <TableHead>
                        {// Loop over the header rows
                            headerGroups.map(headerGroup => (
                                // Apply the header row props
                                <TableRow {...headerGroup.getHeaderGroupProps()}>
                                    {// Loop over the headers in each row
                                        headerGroup.headers.map(column => (
                                            // Apply the header cell props
                                            <TableCell {...column.getHeaderProps()}>
                                                {// Render the header
                                                    column.render('Header')}
                                            </TableCell>
                                        ))}
                                </TableRow>
                            ))}
                    </TableHead>
                    {/* Apply the table body props */}
                    <TableBody {...getTableBodyProps()}>
                        {// Loop over the table rows
                            rows.map(row => {
                                // Prepare the row for display
                                prepareRow(row)
                                return (
                                    // Apply the row props
                                    <TableRow {...row.getRowProps()} className="tablecell">
                                        {// Loop over the rows cells
                                            row.cells.map(cell => {
                                                // Apply the cell props
                                                if (cell.column.Header === "Visible") {
                                                    return (
                                                        <TableCell padding="checkbox">
                                                            <Checkbox
                                                                // checked={isItemSelected}
                                                                // inputProps={{ 'aria-labelledby': labelId }}
                                                            />
                                                        </TableCell>

                                                    )
                                                } else {
                                                    return (
                                                        <TableCell {...cell.getCellProps()}>

                                                            {cell.render('Cell')}

                                                        </TableCell>
                                                    )
                                                }
                                            })}
                                    </TableRow>
                                )
                            })}
                    </TableBody>
                </MaUTable>
            </ThemeProvider>
        </div>

    )


}

export {MetricTable}