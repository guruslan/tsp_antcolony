/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#ifndef _TSP_ANTCOLONY_H_
#define _TSP_ANYCOLONY_H_

// Thread block size
#define BLOCK_SIZE 256

#define BLOCKS 16

//Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA 45 // Matrix A width

#endif // _TSP_ANTCOLONY_H_

