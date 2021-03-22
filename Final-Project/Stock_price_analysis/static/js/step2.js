
const t = d3.select("#stock_table");

const csvfile = d3.select("#t_id").property("value");
console.log(csvfile);

d3.csv("../static/data/" + csvfile + ".csv").then(function (data) {
    console.log("data");
    console.log(data);

    data.forEach(function (d) {
        // Table presentaion
        const row = t.append("tr");
        cell = row.append("td");
        cell.text(d.Date);
        cell = row.append("td");
        cell.text(d.Open);
        cell = row.append("td");
        cell.text(d.High);
        cell = row.append("td");
        cell.text(d.Low);
        cell = row.append("td");
        cell.text(d.Close);
        cell = row.append("td");
        cell.text(d.Volumn);
    });
});
