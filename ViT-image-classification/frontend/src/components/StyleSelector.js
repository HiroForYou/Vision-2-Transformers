import React from "react";
import { makeStyles } from "@material-ui/core/styles";

import InputLabel from "@material-ui/core/InputLabel";
import FormControl from "@material-ui/core/FormControl";
import Select from "@material-ui/core/Select";
import MenuItem from "@material-ui/core/MenuItem";

import { transform } from "../api.js";

const useStyles = makeStyles((theme) => ({
  formControl: {
    width: "70%",
    margin: theme.spacing(1),
  },
}));

export default function StyleSelector(props) {
  const classes = useStyles();

  return (
    <FormControl className={classes.formControl}>
      <InputLabel htmlFor="age-native-helper">
        Select a transformer model
      </InputLabel>
      <Select
        value={props.modelID}
        onChange={(event) => {
          props.setModelID(event.target.value);
          props.setPercentage(1);
          props.setOpen(true);

          const data = {
            image: props.before,
            model_id: event.target.value,
          };

          transform(data)
            .then((response) => {
              console.log("success");
              console.log(response.data);
              props.setPercentage(0);
              props.setOpen(false);
            })
            .catch((response) => {
              console.log(response);
            });
        }}
        inputProps={{
          name: "age",
          id: "age-native-helper",
        }}
      >
        <MenuItem value={0}>B_16_imagenet1k</MenuItem>
        <MenuItem value={1}>B_32_imagenet1k</MenuItem>
      </Select>
    </FormControl>
  );
}
