
// Creating our plot
d3.json("/plots").then(response => {
    console.log(data)
  
  var trace1 = {
    x: response.x,
    y: response.y,
    name: 'MGW',
    type: 'bar'
  };
  
  var data = [trace1];
    
  var layout = {
    title: 'Your Hourly Energy Consumption',
    xaxis: {
    title:  "Range of Date & Hours"
       },
    yaxis: {
      title: "MGW"
     }
  };

  Plotly.newPlot('bar', data, layout);

  
// Creating our table
  tbody=d3.select("tbody");
  tbody.html("");
  for (var i = 0; i < response.x.length; i++){
    var row = tbody.append("tr");
    row.append("td").text(response.x[i])
    row.append("td").text(response.y[i])
  }
});


