using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Security;
using UnityEngine;

public class DebugRBSkeleton : MonoBehaviour
{
    [SerializeField] private int framerate;

    private float timer;
    private float frameTime;

    private Dictionary<string, Dictionary<string, Transform>> skeletonList;

    private void Awake()
    {
        frameTime = 1f / framerate;
        timer = 0;
        skeletonList = new Dictionary<string, Dictionary<string, Transform>>();
    }

    void Update()
    {
        while (PythonLauncher.data.Count > 0)
        {
            if (timer > 0)
            {
                timer -= Time.deltaTime;
                return;
            }
            try
            {
                // Try consume data
                HandlePythonData(PythonLauncher.data[0]);
                PythonLauncher.data.RemoveAt(0);
            }
            catch (Exception e)
            {
                Debug.LogWarning("Something went wrong with: " + PythonLauncher.data[0] + "\n" + e.Message + "\n" + e.StackTrace);
                PythonLauncher.data.RemoveAt(0);
                throw e;
            }
        }
    }

    void HandlePythonData(string data)
    {
        string[] info = data.Split(' '); //Split at whitespace

        switch (info[0]) //First element describes data-type
        {
            case "H":
                ProcessHierarchyDefinition(info);
                break;
            case "G":
                ProcessGlobalJointData(info);
                break;
            case "E":
                timer = frameTime;
                break;
        }
    }

    void ProcessHierarchyDefinition(string[] unparsedData)
    {
        string hierarchyName = unparsedData[1]; //first index is hierarchy name
        // float hipSpineOffset = ParseFloat(unparsedData[2]);

        var skeletonDefinition = new Dictionary<string, Transform>();
        GameObject hierarchyParent = GameObject.Find($@"model_{hierarchyName}");

        //Rest of the data are parent-child pairs by names, so here we simply pair them with Transforms on the skeleton
        for (uint pairIndex = 2; pairIndex < unparsedData.Length; pairIndex++)
        {
            //Pair format: "<child_name>-<parent_name>"
            string[] childParentPair = unparsedData[pairIndex].Split('-');
            string childName = childParentPair[0];

            if (!skeletonDefinition.ContainsKey(childName))
            {
                var childList = hierarchyParent.GetComponentsInChildren<Transform>();
                Transform childTransform = null;
                foreach (Transform child in childList)
                {
                    if (child.gameObject.name == $@"Model:{childName}") childTransform = child;
                }

                if (childTransform != null)
                {
                    skeletonDefinition.Add(childName, childTransform);
                }
            }
        }
        
        skeletonList.Add(hierarchyName, skeletonDefinition);
    }

    void ProcessGlobalJointData(string[] info)
    {
        string skeletonName = info[1];
        string jointName = info[2];

        Vector3 jointPos = new Vector3(
            -ParseFloat(info[3]), 
            ParseFloat(info[4]),
            ParseFloat(info[5])
            );

        Quaternion jointQuat = new Quaternion(
            ParseFloat(info[6]), 
            -ParseFloat(info[7]), 
            -ParseFloat(info[8]),
            ParseFloat(info[9])
            );

        // if (skeletonName == "original")
        // {
        //     jointPos += Vector3.left * _skeletonSpacing;
        // }
        
        UpdateJoint(skeletonName, jointName, jointPos, jointQuat);
    }

    void UpdateJoint(string skeletonName, string jointName, Vector3 position, Quaternion quat)
    {
        if(jointName == "Hips")
            skeletonList[skeletonName][jointName].localPosition += position;   

        skeletonList[skeletonName][jointName].localRotation = quat;
    }

    float ParseFloat(string s)
    {
        return float.Parse(s, CultureInfo.InvariantCulture.NumberFormat);
    }
}
