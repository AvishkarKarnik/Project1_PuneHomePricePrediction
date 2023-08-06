function getBathValue() {
  var uiBathrooms = document.getElementsByName("uiBathrooms");
  for(var i in uiBathrooms) {
    if(uiBathrooms[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}

function getBHKValue() {
  var uiBHK = document.getElementsByName("uiBHK");
  for(var i in uiBHK) {
    if(uiBHK[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}

function getBalconyValue() {
  var uiBalcony = document.getElementsByName("uiBalcony");
  for(var i in uiBalcony) {
    if(uiBalcony[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}

function getAreaType(){
  var uiAreaType = document.getElementById("uiAreaType")
  return uiAreaType.options[uiAreaType.selectedIndex].value
}

// var b = document.querySelector("#button");
// if (b) {
//   b.addEventListener("click", onClickedEstimatePrice); // Remove the parentheses here
// }

function onClickedEstimatePrice() {
  console.log("Estimate price button clicked");
  var sqft = document.getElementById("uiSqft");
  var bhk = getBHKValue();
  var bathrooms = getBathValue();
  var balcony = getBalconyValue();
  var location = document.getElementById("uiLocations");
  var area_type = getAreaType();
  var estPrice = document.getElementById("uiEstimatedPrice");

  var url = "http://127.0.0.1:5000/predict_price"; 
//  var url = "/api/predict_price";

  $.post(url, {
      total_sqft: parseFloat(sqft.value),
      bhk: bhk,
      bath: bathrooms,
      balcony: balcony,
      location: location.value,
      area_type: area_type,
  },
  function(data, status) {
      console.log(data.estimated_price);
      estPrice.innerHTML = "<h2>" + data.estimated_price + " Lakhs</h2>";
      console.log(0);
});
}

function onPageLoad() {
  console.log("document loaded" );
  var url = "http://127.0.0.1:5000/get_location_names"; 
  // var url = "/api/get_location_names";
  $.get(url,function(data, status) {
      console.log("got response for get_location_names request");
      if(data) {
          var locations = data.locations;
          var uiLocations = document.getElementById("uiLocations");
          $('#uiLocations').empty();
          for(var i in locations) {
              var opt = new Option(locations[i]);
              $('#uiLocations').append(opt);
          }
      }
  });
}



window.onload = onPageLoad;

