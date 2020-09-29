unit testmultiarray;

{$mode objfpc}{$H+}

{$IFDEF FPC}
{$PACKRECORDS C}
{$ENDIF}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, multiarray, numerik, DateUtils;

type

  TTestMultiArray = class(TTestCase)
  published
    procedure TestAssignment;
    procedure TestBroadcastMatrixScalar;
    procedure TestBroadcastReshaped;
    procedure TestSlicing2D;
    procedure TestSlicing2DAddScalar;
    procedure TestSlicing2DAddTransposed;
    procedure TestTranspose;

    procedure TestReduceSimple;
    procedure TestReduceStacked;
    procedure TestReduceBig;

    procedure TestVstack;
  end;

var
  M, N: TMultiArray;

implementation

procedure TTestMultiArray.TestAssignment;
var
  t: TDateTime;
begin
  t := Now;
  Write('Generating a 10000x1000 array... ');
  M := Random([10000, 1000]);
  WriteLn(MilliSecondsBetween(Now, t), 'ms');
  t := Now;
  Write('Copying array... ');
  N := M.Copy();
  WriteLn(MilliSecondsBetween(Now, t), 'ms');
end;

procedure TTestMultiArray.TestBroadcastMatrixScalar;
begin
  M := TMultiArray([1, 2, 3, 4]).Reshape([2, 2]);
  AssertTrue(VectorEqual((M + 2).GetVirtualData, [3, 4, 5, 6]));
end;

procedure TTestMultiArray.TestBroadcastReshaped;
begin
  M := [1, 2, 3];
  M := M.Reshape([3, 1]) * M.Reshape([1, 3]);

  AssertTrue(VectorEqual(M.GetVirtualData, [1, 2, 3, 2, 4, 6, 3, 6, 9]));
end;

procedure TTestMultiArray.TestSlicing2D;
var
  M: TMultiArray;
begin
  M := TMultiArray(TSingleVector(Range(1, 10))).Reshape([3, 3]);
  M := M.Slice([[0, 2], [1]]); // ==> [[2], [8]]
  N := TMultiArray([2, 8]).Reshape([2, 1]);
  Assert(ArrayEqual(M, N));
end;

procedure TTestMultiArray.TestSlicing2DAddScalar;
begin
  M := TMultiArray(TSingleVector(Range(1, 10))).Reshape([3, 3]);
  M := M.Slice([[0, 2], [1]]) + 10;
  Assert(VectorEqual(M.GetVirtualData, [12, 18]));
end;

procedure TTestMultiArray.TestSlicing2DAddTransposed;
begin
  M := TMultiArray(TSingleVector(Range(1, 10))).Reshape([3, 3]);
  M := M.Slice([[0, 2], [1]]);
  M := M.T + M;
  AssertTrue(VectorEqual(M.GetVirtualData, [4, 10, 10, 16]));
end;

procedure TTestMultiArray.TestTranspose;
begin
  M := [1, 2, 3, 4, 5, 6];
  M := M.Reshape([2, 3]);
  N := [1, 4, 2, 5, 3, 6];
  N := N.Reshape([3, 2]);
  AssertTrue(ArrayEqual(M.T, N));
end;

procedure TTestMultiArray.TestReduceSimple;
begin
  M := TMultiArray([1, 2, 3, 4, 5, 6]).Reshape([2, 3]);
  M := Sum(M, 0);
  AssertTrue(VectorEqual(M.GetVirtualData, [5, 7, 9]));
end;

procedure TTestMultiArray.TestReduceStacked;
begin
  M := TMultiArray([1, 2, 3, 4, 5, 6]).Reshape([2, 3]);
  M := Sum(Sum(M, 0), 0);
  AssertTrue(VectorEqual(M.GetVirtualData, [21]));
end;

procedure TTestMultiArray.TestReduceBig;
var
  i: integer;
  expected: TSingleVector;
  t: TDateTime;
begin
  WriteLn('Generating a 10000x1000 array... ');
  M := Random([10000, 1000]);
  M := M / M;

  t := Now;
  WriteLn('Reducing array... ');
  M := Sum(M, 0);
  WriteLn(MilliSecondsBetween(Now, t), 'ms');

  SetLength(expected, 1000);
  for i := 0 to 999 do
    expected[i] := 10000;

  AssertTrue(VectorEqual(M.GetVirtualData, expected));
end;

procedure TTestMultiArray.TestVstack;
begin
  M := VStack([TMultiArray([1, 2, 3, 4]).Reshape([2, 2]), Ones([2, 2])]);
  N := TMultiArray([1, 2, 3, 4, 1, 1, 1, 1]).Reshape([4, 2]);
  AssertTrue(ArrayEqual(M, N));
end;

initialization
  RegisterTest(TTestMultiArray);

end.


